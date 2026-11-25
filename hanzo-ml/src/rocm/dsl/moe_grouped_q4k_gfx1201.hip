#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
typedef unsigned int uint;
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long int uint64;
typedef signed char int8;
typedef signed short int16;
typedef signed int int32;
typedef signed long long int int64;

struct __align__(16) int32_4 {
    int32 i_0;
    int32 i_1;
    int32 i_2;
    int32 i_3;
};

struct __align__(16) uint32_4 {
    uint32 i_0;
    uint32 i_1;
    uint32 i_2;
    uint32 i_3;
};

struct __align__(4) __half_2 {
    __half i_0;
    __half i_1;
};

struct __align__(4) int8_4 {
    int8 i_0;
    int8 i_1;
    int8 i_2;
    int8 i_3;
};

struct info_st {
    uint32 static_meta[20];
};


extern "C" __global__ void __launch_bounds__(128) moe_grouped_q4k_gfx1201(
    const uint32_4* __restrict__ buffer_0, const uint32_4* __restrict__ buffer_1, const __half* __restrict__ buffer_2, const int32* __restrict__ buffer_3, const int32* __restrict__ buffer_4, const int32* __restrict__ buffer_5, const int32* __restrict__ buffer_6, const int32* __restrict__ buffer_7, const uint32* __restrict__ buffer_8, __half* buffer_9, const info_st* __restrict__ info_ptr
) {
uint32 warpSizeChecked = min(warpSize, blockDim.x * blockDim.y * blockDim.z);
const info_st& info = *info_ptr;
const uint32* dynamic_meta = reinterpret_cast<const uint32*>(
                    reinterpret_cast<const char*>(info_ptr) + sizeof(info_st)
                );

uint32 l_arr_28[4];

float l_arr_29[4];

int32 l_arr_90[4];

int32 l_arr_91[4];

int32 l_arr_92[4];

int32 l_arr_93[4];

uint32 l_arr_1581[1];

float l_arr_1582[1];

int32 l_arr_1631[1];

int32 l_arr_1632[1];

int32 l_arr_1633[1];

int32 l_arr_1634[1];

uint32 l_mut_22;
uint32 l_mut_42;
uint32 l_mut_65;
uint32 l_mut_67;
const uint32 l_0 = buffer_8[uint32(0)];
const uint32 l_1 = buffer_8[uint32(1)];
const uint32 l_2 = buffer_8[uint32(2)];
const uint32 l_3 = blockIdx.x / l_2;
const uint32 l_4 = blockIdx.x % l_2;
const uint32 l_5 = l_4 * uint32(4);
const uint32 l_6 = threadIdx.x / uint32(32);
const uint32 l_7 = l_5 + l_6;
const int32 l_8 = buffer_7[uint32(0)];
const uint32 l_9 = uint32(l_8);
const bool l_10 = l_3 < l_9;
const bool l_11 = l_7 < l_0;
const bool l_12 = l_10 && l_11;
if (l_12) {
const uint32 l_13 = threadIdx.x % uint32(32);
const int32 l_14 = buffer_4[l_3];
const uint32 l_15 = uint32(l_14);
const uint32 l_16 = l_15 * l_0;
const uint32 l_17 = l_16 + l_7;
const int32 l_18 = buffer_5[l_3];
const uint32 l_19 = uint32(l_18);
const int32 l_20 = buffer_6[l_3];
const uint32 l_21 = uint32(l_20);
l_mut_22 = uint32(0);
while (true) {
const uint32 l_23 = l_mut_22 + uint32(4);
const bool l_24 = l_23 <= l_21;
const bool l_25 = !l_24;
if (l_25) {
break;}
const uint32 l_26 = l_19 + l_mut_22;
const uint32 l_27 = l_1 / uint32(256);
const uint32 l_30 = l_26 + uint32(0);
const int32 l_31 = buffer_3[l_30];
const uint32 l_32 = uint32(l_31);
l_arr_28[uint32(0)] = l_32;
l_arr_29[uint32(0)] = float(0.0);
const uint32 l_33 = l_26 + uint32(1);
const int32 l_34 = buffer_3[l_33];
const uint32 l_35 = uint32(l_34);
l_arr_28[uint32(1)] = l_35;
l_arr_29[uint32(1)] = float(0.0);
const uint32 l_36 = l_26 + uint32(2);
const int32 l_37 = buffer_3[l_36];
const uint32 l_38 = uint32(l_37);
l_arr_28[uint32(2)] = l_38;
l_arr_29[uint32(2)] = float(0.0);
const uint32 l_39 = l_26 + uint32(3);
const int32 l_40 = buffer_3[l_39];
const uint32 l_41 = uint32(l_40);
l_arr_28[uint32(3)] = l_41;
l_arr_29[uint32(3)] = float(0.0);
l_mut_42 = l_13;
while (true) {
const uint32 l_43 = l_27 * uint32(4);
const bool l_44 = l_mut_42 < l_43;
const bool l_45 = !l_44;
if (l_45) {
break;}
const uint32 l_46 = l_mut_42 / uint32(4);
const uint32 l_47 = l_mut_42 % uint32(4);
const uint32 l_48 = l_17 * l_27;
const uint32 l_49 = l_48 + l_46;
const uint32 l_50 = l_49 * uint32(9);
const uint32_4 l_51 = buffer_0[l_50];
const uint32 l_52 = l_51.i_0;
const __half_2 l_53 = reinterpret_cast<__half_2 const&>(l_52);
const __half l_54 = l_53.i_0;
const float l_55 = float(l_54);
const __half l_56 = l_53.i_1;
const float l_57 = float(l_56);
const uint32 l_58 = l_47 % uint32(2);
const uint32 l_59 = l_58 * uint32(16);
const uint32 l_60 = l_51.i_1;
const uint32 l_61 = l_60 >> l_59;
const uint32 l_62 = l_51.i_2;
const uint32 l_63 = l_62 >> l_59;
const uint32 l_64 = l_61 & uint32(16191);
l_mut_65 = l_64;
const uint32 l_66 = l_63 & uint32(16191);
l_mut_67 = l_66;
const bool l_68 = l_47 >= uint32(2);
if (l_68) {
const uint32 l_69 = l_51.i_3;
const uint32 l_70 = l_69 >> l_59;
const uint32 l_71 = l_70 & uint32(3855);
const uint32 l_72 = l_61 >> uint32(2);
const uint32 l_73 = l_72 & uint32(12336);
const uint32 l_74 = l_71 | l_73;
l_mut_65 = l_74;
const uint32 l_75 = l_70 >> uint32(4);
const uint32 l_76 = l_75 & uint32(3855);
const uint32 l_77 = l_63 >> uint32(2);
const uint32 l_78 = l_77 & uint32(12336);
const uint32 l_79 = l_76 | l_78;
l_mut_67 = l_79;
}
const uint32 l_80 = l_mut_65 & uint32(63);
const int32 l_81 = int32(l_80);
const uint32 l_82 = l_mut_65 >> uint32(8);
const uint32 l_83 = l_82 & uint32(63);
const int32 l_84 = int32(l_83);
const uint32 l_85 = l_mut_67 & uint32(63);
const int32 l_86 = int32(l_85);
const uint32 l_87 = l_mut_67 >> uint32(8);
const uint32 l_88 = l_87 & uint32(63);
const int32 l_89 = int32(l_88);
l_arr_90[uint32(0)] = int32(0);
l_arr_91[uint32(0)] = int32(0);
l_arr_92[uint32(0)] = int32(0);
l_arr_93[uint32(0)] = int32(0);
l_arr_90[uint32(1)] = int32(0);
l_arr_91[uint32(1)] = int32(0);
l_arr_92[uint32(1)] = int32(0);
l_arr_93[uint32(1)] = int32(0);
l_arr_90[uint32(2)] = int32(0);
l_arr_91[uint32(2)] = int32(0);
l_arr_92[uint32(2)] = int32(0);
l_arr_93[uint32(2)] = int32(0);
l_arr_90[uint32(3)] = int32(0);
l_arr_91[uint32(3)] = int32(0);
l_arr_92[uint32(3)] = int32(0);
l_arr_93[uint32(3)] = int32(0);
const uint32 l_94 = l_50 + uint32(1);
const uint32 l_95 = l_47 * uint32(2);
const uint32 l_96 = l_94 + l_95;
const uint32 l_97 = l_96 + uint32(0);
const uint32_4 l_98 = buffer_0[l_97];
const uint32 l_99 = l_arr_28[uint32(0)];
const uint32 l_100 = l_1 / uint32(16);
const uint32 l_101 = l_99 * l_100;
const uint32 l_102 = l_46 * uint32(16);
const uint32 l_103 = l_101 + l_102;
const uint32 l_104 = l_47 * uint32(4);
const uint32 l_105 = l_103 + l_104;
const uint32 l_106 = l_105 + uint32(0);
const uint32_4 l_107 = buffer_1[l_106];
const uint32 l_108 = l_106 + uint32(2);
const uint32_4 l_109 = buffer_1[l_108];
const uint32 l_110 = l_98.i_0;
const uint32 l_111 = l_110 & uint32(252645135);
const uint32 l_112 = l_110 >> uint32(4);
const uint32 l_113 = l_112 & uint32(252645135);
const uint32 l_114 = l_107.i_0;
const int8_4 l_115 = reinterpret_cast<int8_4 const&>(l_111);
const int32_4 l_116 = int32_4{
int32(l_115.i_0),int32(l_115.i_1),int32(l_115.i_2),int32(l_115.i_3),};
const int8_4 l_117 = reinterpret_cast<int8_4 const&>(l_114);
const int32_4 l_118 = int32_4{
int32(l_117.i_0),int32(l_117.i_1),int32(l_117.i_2),int32(l_117.i_3),};
#if defined(__gfx1201__)
const int32 l_119 = __builtin_amdgcn_sudot4(true, l_111, true, l_114, 0, false);
#else
const int32 l_119 = l_116.i_0 * l_118.i_0 + l_116.i_1 * l_118.i_1 + l_116.i_2 * l_118.i_2 + l_116.i_3 * l_118.i_3;
#endif
const int32 l_120 = l_arr_90[uint32(0)];
const int32 l_121 = l_120 + l_119;
l_arr_90[uint32(0)] = l_121;
const uint32 l_122 = l_109.i_0;
const int8_4 l_123 = reinterpret_cast<int8_4 const&>(l_113);
const int32_4 l_124 = int32_4{
int32(l_123.i_0),int32(l_123.i_1),int32(l_123.i_2),int32(l_123.i_3),};
const int8_4 l_125 = reinterpret_cast<int8_4 const&>(l_122);
const int32_4 l_126 = int32_4{
int32(l_125.i_0),int32(l_125.i_1),int32(l_125.i_2),int32(l_125.i_3),};
#if defined(__gfx1201__)
const int32 l_127 = __builtin_amdgcn_sudot4(true, l_113, true, l_122, 0, false);
#else
const int32 l_127 = l_124.i_0 * l_126.i_0 + l_124.i_1 * l_126.i_1 + l_124.i_2 * l_126.i_2 + l_124.i_3 * l_126.i_3;
#endif
const int32 l_128 = l_arr_91[uint32(0)];
const int32 l_129 = l_128 + l_127;
l_arr_91[uint32(0)] = l_129;
const uint32 l_130 = l_107.i_0;
const uint32 l_131_bits = uint32(16843009);
const int8_4 l_131 = reinterpret_cast<int8_4 const&>(l_131_bits);
const int32_4 l_132 = int32_4{
int32(l_131.i_0),int32(l_131.i_1),int32(l_131.i_2),int32(l_131.i_3),};
const int8_4 l_133 = reinterpret_cast<int8_4 const&>(l_130);
const int32_4 l_134 = int32_4{
int32(l_133.i_0),int32(l_133.i_1),int32(l_133.i_2),int32(l_133.i_3),};
#if defined(__gfx1201__)
const int32 l_135 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_130, 0, false);
#else
const int32 l_135 = l_132.i_0 * l_134.i_0 + l_132.i_1 * l_134.i_1 + l_132.i_2 * l_134.i_2 + l_132.i_3 * l_134.i_3;
#endif
const int32 l_136 = l_arr_92[uint32(0)];
const int32 l_137 = l_136 + l_135;
l_arr_92[uint32(0)] = l_137;
const uint32 l_138 = l_109.i_0;
const uint32 l_139_bits = uint32(16843009);
const int8_4 l_139 = reinterpret_cast<int8_4 const&>(l_139_bits);
const int32_4 l_140 = int32_4{
int32(l_139.i_0),int32(l_139.i_1),int32(l_139.i_2),int32(l_139.i_3),};
const int8_4 l_141 = reinterpret_cast<int8_4 const&>(l_138);
const int32_4 l_142 = int32_4{
int32(l_141.i_0),int32(l_141.i_1),int32(l_141.i_2),int32(l_141.i_3),};
#if defined(__gfx1201__)
const int32 l_143 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_138, 0, false);
#else
const int32 l_143 = l_140.i_0 * l_142.i_0 + l_140.i_1 * l_142.i_1 + l_140.i_2 * l_142.i_2 + l_140.i_3 * l_142.i_3;
#endif
const int32 l_144 = l_arr_93[uint32(0)];
const int32 l_145 = l_144 + l_143;
l_arr_93[uint32(0)] = l_145;
const uint32 l_146 = l_98.i_1;
const uint32 l_147 = l_146 & uint32(252645135);
const uint32 l_148 = l_146 >> uint32(4);
const uint32 l_149 = l_148 & uint32(252645135);
const uint32 l_150 = l_107.i_1;
const int8_4 l_151 = reinterpret_cast<int8_4 const&>(l_147);
const int32_4 l_152 = int32_4{
int32(l_151.i_0),int32(l_151.i_1),int32(l_151.i_2),int32(l_151.i_3),};
const int8_4 l_153 = reinterpret_cast<int8_4 const&>(l_150);
const int32_4 l_154 = int32_4{
int32(l_153.i_0),int32(l_153.i_1),int32(l_153.i_2),int32(l_153.i_3),};
#if defined(__gfx1201__)
const int32 l_155 = __builtin_amdgcn_sudot4(true, l_147, true, l_150, 0, false);
#else
const int32 l_155 = l_152.i_0 * l_154.i_0 + l_152.i_1 * l_154.i_1 + l_152.i_2 * l_154.i_2 + l_152.i_3 * l_154.i_3;
#endif
const int32 l_156 = l_arr_90[uint32(0)];
const int32 l_157 = l_156 + l_155;
l_arr_90[uint32(0)] = l_157;
const uint32 l_158 = l_109.i_1;
const int8_4 l_159 = reinterpret_cast<int8_4 const&>(l_149);
const int32_4 l_160 = int32_4{
int32(l_159.i_0),int32(l_159.i_1),int32(l_159.i_2),int32(l_159.i_3),};
const int8_4 l_161 = reinterpret_cast<int8_4 const&>(l_158);
const int32_4 l_162 = int32_4{
int32(l_161.i_0),int32(l_161.i_1),int32(l_161.i_2),int32(l_161.i_3),};
#if defined(__gfx1201__)
const int32 l_163 = __builtin_amdgcn_sudot4(true, l_149, true, l_158, 0, false);
#else
const int32 l_163 = l_160.i_0 * l_162.i_0 + l_160.i_1 * l_162.i_1 + l_160.i_2 * l_162.i_2 + l_160.i_3 * l_162.i_3;
#endif
const int32 l_164 = l_arr_91[uint32(0)];
const int32 l_165 = l_164 + l_163;
l_arr_91[uint32(0)] = l_165;
const uint32 l_166 = l_107.i_1;
const uint32 l_167_bits = uint32(16843009);
const int8_4 l_167 = reinterpret_cast<int8_4 const&>(l_167_bits);
const int32_4 l_168 = int32_4{
int32(l_167.i_0),int32(l_167.i_1),int32(l_167.i_2),int32(l_167.i_3),};
const int8_4 l_169 = reinterpret_cast<int8_4 const&>(l_166);
const int32_4 l_170 = int32_4{
int32(l_169.i_0),int32(l_169.i_1),int32(l_169.i_2),int32(l_169.i_3),};
#if defined(__gfx1201__)
const int32 l_171 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_166, 0, false);
#else
const int32 l_171 = l_168.i_0 * l_170.i_0 + l_168.i_1 * l_170.i_1 + l_168.i_2 * l_170.i_2 + l_168.i_3 * l_170.i_3;
#endif
const int32 l_172 = l_arr_92[uint32(0)];
const int32 l_173 = l_172 + l_171;
l_arr_92[uint32(0)] = l_173;
const uint32 l_174 = l_109.i_1;
const uint32 l_175_bits = uint32(16843009);
const int8_4 l_175 = reinterpret_cast<int8_4 const&>(l_175_bits);
const int32_4 l_176 = int32_4{
int32(l_175.i_0),int32(l_175.i_1),int32(l_175.i_2),int32(l_175.i_3),};
const int8_4 l_177 = reinterpret_cast<int8_4 const&>(l_174);
const int32_4 l_178 = int32_4{
int32(l_177.i_0),int32(l_177.i_1),int32(l_177.i_2),int32(l_177.i_3),};
#if defined(__gfx1201__)
const int32 l_179 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_174, 0, false);
#else
const int32 l_179 = l_176.i_0 * l_178.i_0 + l_176.i_1 * l_178.i_1 + l_176.i_2 * l_178.i_2 + l_176.i_3 * l_178.i_3;
#endif
const int32 l_180 = l_arr_93[uint32(0)];
const int32 l_181 = l_180 + l_179;
l_arr_93[uint32(0)] = l_181;
const uint32 l_182 = l_98.i_2;
const uint32 l_183 = l_182 & uint32(252645135);
const uint32 l_184 = l_182 >> uint32(4);
const uint32 l_185 = l_184 & uint32(252645135);
const uint32 l_186 = l_107.i_2;
const int8_4 l_187 = reinterpret_cast<int8_4 const&>(l_183);
const int32_4 l_188 = int32_4{
int32(l_187.i_0),int32(l_187.i_1),int32(l_187.i_2),int32(l_187.i_3),};
const int8_4 l_189 = reinterpret_cast<int8_4 const&>(l_186);
const int32_4 l_190 = int32_4{
int32(l_189.i_0),int32(l_189.i_1),int32(l_189.i_2),int32(l_189.i_3),};
#if defined(__gfx1201__)
const int32 l_191 = __builtin_amdgcn_sudot4(true, l_183, true, l_186, 0, false);
#else
const int32 l_191 = l_188.i_0 * l_190.i_0 + l_188.i_1 * l_190.i_1 + l_188.i_2 * l_190.i_2 + l_188.i_3 * l_190.i_3;
#endif
const int32 l_192 = l_arr_90[uint32(0)];
const int32 l_193 = l_192 + l_191;
l_arr_90[uint32(0)] = l_193;
const uint32 l_194 = l_109.i_2;
const int8_4 l_195 = reinterpret_cast<int8_4 const&>(l_185);
const int32_4 l_196 = int32_4{
int32(l_195.i_0),int32(l_195.i_1),int32(l_195.i_2),int32(l_195.i_3),};
const int8_4 l_197 = reinterpret_cast<int8_4 const&>(l_194);
const int32_4 l_198 = int32_4{
int32(l_197.i_0),int32(l_197.i_1),int32(l_197.i_2),int32(l_197.i_3),};
#if defined(__gfx1201__)
const int32 l_199 = __builtin_amdgcn_sudot4(true, l_185, true, l_194, 0, false);
#else
const int32 l_199 = l_196.i_0 * l_198.i_0 + l_196.i_1 * l_198.i_1 + l_196.i_2 * l_198.i_2 + l_196.i_3 * l_198.i_3;
#endif
const int32 l_200 = l_arr_91[uint32(0)];
const int32 l_201 = l_200 + l_199;
l_arr_91[uint32(0)] = l_201;
const uint32 l_202 = l_107.i_2;
const uint32 l_203_bits = uint32(16843009);
const int8_4 l_203 = reinterpret_cast<int8_4 const&>(l_203_bits);
const int32_4 l_204 = int32_4{
int32(l_203.i_0),int32(l_203.i_1),int32(l_203.i_2),int32(l_203.i_3),};
const int8_4 l_205 = reinterpret_cast<int8_4 const&>(l_202);
const int32_4 l_206 = int32_4{
int32(l_205.i_0),int32(l_205.i_1),int32(l_205.i_2),int32(l_205.i_3),};
#if defined(__gfx1201__)
const int32 l_207 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_202, 0, false);
#else
const int32 l_207 = l_204.i_0 * l_206.i_0 + l_204.i_1 * l_206.i_1 + l_204.i_2 * l_206.i_2 + l_204.i_3 * l_206.i_3;
#endif
const int32 l_208 = l_arr_92[uint32(0)];
const int32 l_209 = l_208 + l_207;
l_arr_92[uint32(0)] = l_209;
const uint32 l_210 = l_109.i_2;
const uint32 l_211_bits = uint32(16843009);
const int8_4 l_211 = reinterpret_cast<int8_4 const&>(l_211_bits);
const int32_4 l_212 = int32_4{
int32(l_211.i_0),int32(l_211.i_1),int32(l_211.i_2),int32(l_211.i_3),};
const int8_4 l_213 = reinterpret_cast<int8_4 const&>(l_210);
const int32_4 l_214 = int32_4{
int32(l_213.i_0),int32(l_213.i_1),int32(l_213.i_2),int32(l_213.i_3),};
#if defined(__gfx1201__)
const int32 l_215 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_210, 0, false);
#else
const int32 l_215 = l_212.i_0 * l_214.i_0 + l_212.i_1 * l_214.i_1 + l_212.i_2 * l_214.i_2 + l_212.i_3 * l_214.i_3;
#endif
const int32 l_216 = l_arr_93[uint32(0)];
const int32 l_217 = l_216 + l_215;
l_arr_93[uint32(0)] = l_217;
const uint32 l_218 = l_98.i_3;
const uint32 l_219 = l_218 & uint32(252645135);
const uint32 l_220 = l_218 >> uint32(4);
const uint32 l_221 = l_220 & uint32(252645135);
const uint32 l_222 = l_107.i_3;
const int8_4 l_223 = reinterpret_cast<int8_4 const&>(l_219);
const int32_4 l_224 = int32_4{
int32(l_223.i_0),int32(l_223.i_1),int32(l_223.i_2),int32(l_223.i_3),};
const int8_4 l_225 = reinterpret_cast<int8_4 const&>(l_222);
const int32_4 l_226 = int32_4{
int32(l_225.i_0),int32(l_225.i_1),int32(l_225.i_2),int32(l_225.i_3),};
#if defined(__gfx1201__)
const int32 l_227 = __builtin_amdgcn_sudot4(true, l_219, true, l_222, 0, false);
#else
const int32 l_227 = l_224.i_0 * l_226.i_0 + l_224.i_1 * l_226.i_1 + l_224.i_2 * l_226.i_2 + l_224.i_3 * l_226.i_3;
#endif
const int32 l_228 = l_arr_90[uint32(0)];
const int32 l_229 = l_228 + l_227;
l_arr_90[uint32(0)] = l_229;
const uint32 l_230 = l_109.i_3;
const int8_4 l_231 = reinterpret_cast<int8_4 const&>(l_221);
const int32_4 l_232 = int32_4{
int32(l_231.i_0),int32(l_231.i_1),int32(l_231.i_2),int32(l_231.i_3),};
const int8_4 l_233 = reinterpret_cast<int8_4 const&>(l_230);
const int32_4 l_234 = int32_4{
int32(l_233.i_0),int32(l_233.i_1),int32(l_233.i_2),int32(l_233.i_3),};
#if defined(__gfx1201__)
const int32 l_235 = __builtin_amdgcn_sudot4(true, l_221, true, l_230, 0, false);
#else
const int32 l_235 = l_232.i_0 * l_234.i_0 + l_232.i_1 * l_234.i_1 + l_232.i_2 * l_234.i_2 + l_232.i_3 * l_234.i_3;
#endif
const int32 l_236 = l_arr_91[uint32(0)];
const int32 l_237 = l_236 + l_235;
l_arr_91[uint32(0)] = l_237;
const uint32 l_238 = l_107.i_3;
const uint32 l_239_bits = uint32(16843009);
const int8_4 l_239 = reinterpret_cast<int8_4 const&>(l_239_bits);
const int32_4 l_240 = int32_4{
int32(l_239.i_0),int32(l_239.i_1),int32(l_239.i_2),int32(l_239.i_3),};
const int8_4 l_241 = reinterpret_cast<int8_4 const&>(l_238);
const int32_4 l_242 = int32_4{
int32(l_241.i_0),int32(l_241.i_1),int32(l_241.i_2),int32(l_241.i_3),};
#if defined(__gfx1201__)
const int32 l_243 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_238, 0, false);
#else
const int32 l_243 = l_240.i_0 * l_242.i_0 + l_240.i_1 * l_242.i_1 + l_240.i_2 * l_242.i_2 + l_240.i_3 * l_242.i_3;
#endif
const int32 l_244 = l_arr_92[uint32(0)];
const int32 l_245 = l_244 + l_243;
l_arr_92[uint32(0)] = l_245;
const uint32 l_246 = l_109.i_3;
const uint32 l_247_bits = uint32(16843009);
const int8_4 l_247 = reinterpret_cast<int8_4 const&>(l_247_bits);
const int32_4 l_248 = int32_4{
int32(l_247.i_0),int32(l_247.i_1),int32(l_247.i_2),int32(l_247.i_3),};
const int8_4 l_249 = reinterpret_cast<int8_4 const&>(l_246);
const int32_4 l_250 = int32_4{
int32(l_249.i_0),int32(l_249.i_1),int32(l_249.i_2),int32(l_249.i_3),};
#if defined(__gfx1201__)
const int32 l_251 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_246, 0, false);
#else
const int32 l_251 = l_248.i_0 * l_250.i_0 + l_248.i_1 * l_250.i_1 + l_248.i_2 * l_250.i_2 + l_248.i_3 * l_250.i_3;
#endif
const int32 l_252 = l_arr_93[uint32(0)];
const int32 l_253 = l_252 + l_251;
l_arr_93[uint32(0)] = l_253;
const uint32 l_254 = l_arr_28[uint32(1)];
const uint32 l_255 = l_1 / uint32(16);
const uint32 l_256 = l_254 * l_255;
const uint32 l_257 = l_46 * uint32(16);
const uint32 l_258 = l_256 + l_257;
const uint32 l_259 = l_47 * uint32(4);
const uint32 l_260 = l_258 + l_259;
const uint32 l_261 = l_260 + uint32(0);
const uint32_4 l_262 = buffer_1[l_261];
const uint32 l_263 = l_261 + uint32(2);
const uint32_4 l_264 = buffer_1[l_263];
const uint32 l_265 = l_98.i_0;
const uint32 l_266 = l_265 & uint32(252645135);
const uint32 l_267 = l_265 >> uint32(4);
const uint32 l_268 = l_267 & uint32(252645135);
const uint32 l_269 = l_262.i_0;
const int8_4 l_270 = reinterpret_cast<int8_4 const&>(l_266);
const int32_4 l_271 = int32_4{
int32(l_270.i_0),int32(l_270.i_1),int32(l_270.i_2),int32(l_270.i_3),};
const int8_4 l_272 = reinterpret_cast<int8_4 const&>(l_269);
const int32_4 l_273 = int32_4{
int32(l_272.i_0),int32(l_272.i_1),int32(l_272.i_2),int32(l_272.i_3),};
#if defined(__gfx1201__)
const int32 l_274 = __builtin_amdgcn_sudot4(true, l_266, true, l_269, 0, false);
#else
const int32 l_274 = l_271.i_0 * l_273.i_0 + l_271.i_1 * l_273.i_1 + l_271.i_2 * l_273.i_2 + l_271.i_3 * l_273.i_3;
#endif
const int32 l_275 = l_arr_90[uint32(1)];
const int32 l_276 = l_275 + l_274;
l_arr_90[uint32(1)] = l_276;
const uint32 l_277 = l_264.i_0;
const int8_4 l_278 = reinterpret_cast<int8_4 const&>(l_268);
const int32_4 l_279 = int32_4{
int32(l_278.i_0),int32(l_278.i_1),int32(l_278.i_2),int32(l_278.i_3),};
const int8_4 l_280 = reinterpret_cast<int8_4 const&>(l_277);
const int32_4 l_281 = int32_4{
int32(l_280.i_0),int32(l_280.i_1),int32(l_280.i_2),int32(l_280.i_3),};
#if defined(__gfx1201__)
const int32 l_282 = __builtin_amdgcn_sudot4(true, l_268, true, l_277, 0, false);
#else
const int32 l_282 = l_279.i_0 * l_281.i_0 + l_279.i_1 * l_281.i_1 + l_279.i_2 * l_281.i_2 + l_279.i_3 * l_281.i_3;
#endif
const int32 l_283 = l_arr_91[uint32(1)];
const int32 l_284 = l_283 + l_282;
l_arr_91[uint32(1)] = l_284;
const uint32 l_285 = l_262.i_0;
const uint32 l_286_bits = uint32(16843009);
const int8_4 l_286 = reinterpret_cast<int8_4 const&>(l_286_bits);
const int32_4 l_287 = int32_4{
int32(l_286.i_0),int32(l_286.i_1),int32(l_286.i_2),int32(l_286.i_3),};
const int8_4 l_288 = reinterpret_cast<int8_4 const&>(l_285);
const int32_4 l_289 = int32_4{
int32(l_288.i_0),int32(l_288.i_1),int32(l_288.i_2),int32(l_288.i_3),};
#if defined(__gfx1201__)
const int32 l_290 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_285, 0, false);
#else
const int32 l_290 = l_287.i_0 * l_289.i_0 + l_287.i_1 * l_289.i_1 + l_287.i_2 * l_289.i_2 + l_287.i_3 * l_289.i_3;
#endif
const int32 l_291 = l_arr_92[uint32(1)];
const int32 l_292 = l_291 + l_290;
l_arr_92[uint32(1)] = l_292;
const uint32 l_293 = l_264.i_0;
const uint32 l_294_bits = uint32(16843009);
const int8_4 l_294 = reinterpret_cast<int8_4 const&>(l_294_bits);
const int32_4 l_295 = int32_4{
int32(l_294.i_0),int32(l_294.i_1),int32(l_294.i_2),int32(l_294.i_3),};
const int8_4 l_296 = reinterpret_cast<int8_4 const&>(l_293);
const int32_4 l_297 = int32_4{
int32(l_296.i_0),int32(l_296.i_1),int32(l_296.i_2),int32(l_296.i_3),};
#if defined(__gfx1201__)
const int32 l_298 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_293, 0, false);
#else
const int32 l_298 = l_295.i_0 * l_297.i_0 + l_295.i_1 * l_297.i_1 + l_295.i_2 * l_297.i_2 + l_295.i_3 * l_297.i_3;
#endif
const int32 l_299 = l_arr_93[uint32(1)];
const int32 l_300 = l_299 + l_298;
l_arr_93[uint32(1)] = l_300;
const uint32 l_301 = l_98.i_1;
const uint32 l_302 = l_301 & uint32(252645135);
const uint32 l_303 = l_301 >> uint32(4);
const uint32 l_304 = l_303 & uint32(252645135);
const uint32 l_305 = l_262.i_1;
const int8_4 l_306 = reinterpret_cast<int8_4 const&>(l_302);
const int32_4 l_307 = int32_4{
int32(l_306.i_0),int32(l_306.i_1),int32(l_306.i_2),int32(l_306.i_3),};
const int8_4 l_308 = reinterpret_cast<int8_4 const&>(l_305);
const int32_4 l_309 = int32_4{
int32(l_308.i_0),int32(l_308.i_1),int32(l_308.i_2),int32(l_308.i_3),};
#if defined(__gfx1201__)
const int32 l_310 = __builtin_amdgcn_sudot4(true, l_302, true, l_305, 0, false);
#else
const int32 l_310 = l_307.i_0 * l_309.i_0 + l_307.i_1 * l_309.i_1 + l_307.i_2 * l_309.i_2 + l_307.i_3 * l_309.i_3;
#endif
const int32 l_311 = l_arr_90[uint32(1)];
const int32 l_312 = l_311 + l_310;
l_arr_90[uint32(1)] = l_312;
const uint32 l_313 = l_264.i_1;
const int8_4 l_314 = reinterpret_cast<int8_4 const&>(l_304);
const int32_4 l_315 = int32_4{
int32(l_314.i_0),int32(l_314.i_1),int32(l_314.i_2),int32(l_314.i_3),};
const int8_4 l_316 = reinterpret_cast<int8_4 const&>(l_313);
const int32_4 l_317 = int32_4{
int32(l_316.i_0),int32(l_316.i_1),int32(l_316.i_2),int32(l_316.i_3),};
#if defined(__gfx1201__)
const int32 l_318 = __builtin_amdgcn_sudot4(true, l_304, true, l_313, 0, false);
#else
const int32 l_318 = l_315.i_0 * l_317.i_0 + l_315.i_1 * l_317.i_1 + l_315.i_2 * l_317.i_2 + l_315.i_3 * l_317.i_3;
#endif
const int32 l_319 = l_arr_91[uint32(1)];
const int32 l_320 = l_319 + l_318;
l_arr_91[uint32(1)] = l_320;
const uint32 l_321 = l_262.i_1;
const uint32 l_322_bits = uint32(16843009);
const int8_4 l_322 = reinterpret_cast<int8_4 const&>(l_322_bits);
const int32_4 l_323 = int32_4{
int32(l_322.i_0),int32(l_322.i_1),int32(l_322.i_2),int32(l_322.i_3),};
const int8_4 l_324 = reinterpret_cast<int8_4 const&>(l_321);
const int32_4 l_325 = int32_4{
int32(l_324.i_0),int32(l_324.i_1),int32(l_324.i_2),int32(l_324.i_3),};
#if defined(__gfx1201__)
const int32 l_326 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_321, 0, false);
#else
const int32 l_326 = l_323.i_0 * l_325.i_0 + l_323.i_1 * l_325.i_1 + l_323.i_2 * l_325.i_2 + l_323.i_3 * l_325.i_3;
#endif
const int32 l_327 = l_arr_92[uint32(1)];
const int32 l_328 = l_327 + l_326;
l_arr_92[uint32(1)] = l_328;
const uint32 l_329 = l_264.i_1;
const uint32 l_330_bits = uint32(16843009);
const int8_4 l_330 = reinterpret_cast<int8_4 const&>(l_330_bits);
const int32_4 l_331 = int32_4{
int32(l_330.i_0),int32(l_330.i_1),int32(l_330.i_2),int32(l_330.i_3),};
const int8_4 l_332 = reinterpret_cast<int8_4 const&>(l_329);
const int32_4 l_333 = int32_4{
int32(l_332.i_0),int32(l_332.i_1),int32(l_332.i_2),int32(l_332.i_3),};
#if defined(__gfx1201__)
const int32 l_334 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_329, 0, false);
#else
const int32 l_334 = l_331.i_0 * l_333.i_0 + l_331.i_1 * l_333.i_1 + l_331.i_2 * l_333.i_2 + l_331.i_3 * l_333.i_3;
#endif
const int32 l_335 = l_arr_93[uint32(1)];
const int32 l_336 = l_335 + l_334;
l_arr_93[uint32(1)] = l_336;
const uint32 l_337 = l_98.i_2;
const uint32 l_338 = l_337 & uint32(252645135);
const uint32 l_339 = l_337 >> uint32(4);
const uint32 l_340 = l_339 & uint32(252645135);
const uint32 l_341 = l_262.i_2;
const int8_4 l_342 = reinterpret_cast<int8_4 const&>(l_338);
const int32_4 l_343 = int32_4{
int32(l_342.i_0),int32(l_342.i_1),int32(l_342.i_2),int32(l_342.i_3),};
const int8_4 l_344 = reinterpret_cast<int8_4 const&>(l_341);
const int32_4 l_345 = int32_4{
int32(l_344.i_0),int32(l_344.i_1),int32(l_344.i_2),int32(l_344.i_3),};
#if defined(__gfx1201__)
const int32 l_346 = __builtin_amdgcn_sudot4(true, l_338, true, l_341, 0, false);
#else
const int32 l_346 = l_343.i_0 * l_345.i_0 + l_343.i_1 * l_345.i_1 + l_343.i_2 * l_345.i_2 + l_343.i_3 * l_345.i_3;
#endif
const int32 l_347 = l_arr_90[uint32(1)];
const int32 l_348 = l_347 + l_346;
l_arr_90[uint32(1)] = l_348;
const uint32 l_349 = l_264.i_2;
const int8_4 l_350 = reinterpret_cast<int8_4 const&>(l_340);
const int32_4 l_351 = int32_4{
int32(l_350.i_0),int32(l_350.i_1),int32(l_350.i_2),int32(l_350.i_3),};
const int8_4 l_352 = reinterpret_cast<int8_4 const&>(l_349);
const int32_4 l_353 = int32_4{
int32(l_352.i_0),int32(l_352.i_1),int32(l_352.i_2),int32(l_352.i_3),};
#if defined(__gfx1201__)
const int32 l_354 = __builtin_amdgcn_sudot4(true, l_340, true, l_349, 0, false);
#else
const int32 l_354 = l_351.i_0 * l_353.i_0 + l_351.i_1 * l_353.i_1 + l_351.i_2 * l_353.i_2 + l_351.i_3 * l_353.i_3;
#endif
const int32 l_355 = l_arr_91[uint32(1)];
const int32 l_356 = l_355 + l_354;
l_arr_91[uint32(1)] = l_356;
const uint32 l_357 = l_262.i_2;
const uint32 l_358_bits = uint32(16843009);
const int8_4 l_358 = reinterpret_cast<int8_4 const&>(l_358_bits);
const int32_4 l_359 = int32_4{
int32(l_358.i_0),int32(l_358.i_1),int32(l_358.i_2),int32(l_358.i_3),};
const int8_4 l_360 = reinterpret_cast<int8_4 const&>(l_357);
const int32_4 l_361 = int32_4{
int32(l_360.i_0),int32(l_360.i_1),int32(l_360.i_2),int32(l_360.i_3),};
#if defined(__gfx1201__)
const int32 l_362 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_357, 0, false);
#else
const int32 l_362 = l_359.i_0 * l_361.i_0 + l_359.i_1 * l_361.i_1 + l_359.i_2 * l_361.i_2 + l_359.i_3 * l_361.i_3;
#endif
const int32 l_363 = l_arr_92[uint32(1)];
const int32 l_364 = l_363 + l_362;
l_arr_92[uint32(1)] = l_364;
const uint32 l_365 = l_264.i_2;
const uint32 l_366_bits = uint32(16843009);
const int8_4 l_366 = reinterpret_cast<int8_4 const&>(l_366_bits);
const int32_4 l_367 = int32_4{
int32(l_366.i_0),int32(l_366.i_1),int32(l_366.i_2),int32(l_366.i_3),};
const int8_4 l_368 = reinterpret_cast<int8_4 const&>(l_365);
const int32_4 l_369 = int32_4{
int32(l_368.i_0),int32(l_368.i_1),int32(l_368.i_2),int32(l_368.i_3),};
#if defined(__gfx1201__)
const int32 l_370 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_365, 0, false);
#else
const int32 l_370 = l_367.i_0 * l_369.i_0 + l_367.i_1 * l_369.i_1 + l_367.i_2 * l_369.i_2 + l_367.i_3 * l_369.i_3;
#endif
const int32 l_371 = l_arr_93[uint32(1)];
const int32 l_372 = l_371 + l_370;
l_arr_93[uint32(1)] = l_372;
const uint32 l_373 = l_98.i_3;
const uint32 l_374 = l_373 & uint32(252645135);
const uint32 l_375 = l_373 >> uint32(4);
const uint32 l_376 = l_375 & uint32(252645135);
const uint32 l_377 = l_262.i_3;
const int8_4 l_378 = reinterpret_cast<int8_4 const&>(l_374);
const int32_4 l_379 = int32_4{
int32(l_378.i_0),int32(l_378.i_1),int32(l_378.i_2),int32(l_378.i_3),};
const int8_4 l_380 = reinterpret_cast<int8_4 const&>(l_377);
const int32_4 l_381 = int32_4{
int32(l_380.i_0),int32(l_380.i_1),int32(l_380.i_2),int32(l_380.i_3),};
#if defined(__gfx1201__)
const int32 l_382 = __builtin_amdgcn_sudot4(true, l_374, true, l_377, 0, false);
#else
const int32 l_382 = l_379.i_0 * l_381.i_0 + l_379.i_1 * l_381.i_1 + l_379.i_2 * l_381.i_2 + l_379.i_3 * l_381.i_3;
#endif
const int32 l_383 = l_arr_90[uint32(1)];
const int32 l_384 = l_383 + l_382;
l_arr_90[uint32(1)] = l_384;
const uint32 l_385 = l_264.i_3;
const int8_4 l_386 = reinterpret_cast<int8_4 const&>(l_376);
const int32_4 l_387 = int32_4{
int32(l_386.i_0),int32(l_386.i_1),int32(l_386.i_2),int32(l_386.i_3),};
const int8_4 l_388 = reinterpret_cast<int8_4 const&>(l_385);
const int32_4 l_389 = int32_4{
int32(l_388.i_0),int32(l_388.i_1),int32(l_388.i_2),int32(l_388.i_3),};
#if defined(__gfx1201__)
const int32 l_390 = __builtin_amdgcn_sudot4(true, l_376, true, l_385, 0, false);
#else
const int32 l_390 = l_387.i_0 * l_389.i_0 + l_387.i_1 * l_389.i_1 + l_387.i_2 * l_389.i_2 + l_387.i_3 * l_389.i_3;
#endif
const int32 l_391 = l_arr_91[uint32(1)];
const int32 l_392 = l_391 + l_390;
l_arr_91[uint32(1)] = l_392;
const uint32 l_393 = l_262.i_3;
const uint32 l_394_bits = uint32(16843009);
const int8_4 l_394 = reinterpret_cast<int8_4 const&>(l_394_bits);
const int32_4 l_395 = int32_4{
int32(l_394.i_0),int32(l_394.i_1),int32(l_394.i_2),int32(l_394.i_3),};
const int8_4 l_396 = reinterpret_cast<int8_4 const&>(l_393);
const int32_4 l_397 = int32_4{
int32(l_396.i_0),int32(l_396.i_1),int32(l_396.i_2),int32(l_396.i_3),};
#if defined(__gfx1201__)
const int32 l_398 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_393, 0, false);
#else
const int32 l_398 = l_395.i_0 * l_397.i_0 + l_395.i_1 * l_397.i_1 + l_395.i_2 * l_397.i_2 + l_395.i_3 * l_397.i_3;
#endif
const int32 l_399 = l_arr_92[uint32(1)];
const int32 l_400 = l_399 + l_398;
l_arr_92[uint32(1)] = l_400;
const uint32 l_401 = l_264.i_3;
const uint32 l_402_bits = uint32(16843009);
const int8_4 l_402 = reinterpret_cast<int8_4 const&>(l_402_bits);
const int32_4 l_403 = int32_4{
int32(l_402.i_0),int32(l_402.i_1),int32(l_402.i_2),int32(l_402.i_3),};
const int8_4 l_404 = reinterpret_cast<int8_4 const&>(l_401);
const int32_4 l_405 = int32_4{
int32(l_404.i_0),int32(l_404.i_1),int32(l_404.i_2),int32(l_404.i_3),};
#if defined(__gfx1201__)
const int32 l_406 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_401, 0, false);
#else
const int32 l_406 = l_403.i_0 * l_405.i_0 + l_403.i_1 * l_405.i_1 + l_403.i_2 * l_405.i_2 + l_403.i_3 * l_405.i_3;
#endif
const int32 l_407 = l_arr_93[uint32(1)];
const int32 l_408 = l_407 + l_406;
l_arr_93[uint32(1)] = l_408;
const uint32 l_409 = l_arr_28[uint32(2)];
const uint32 l_410 = l_1 / uint32(16);
const uint32 l_411 = l_409 * l_410;
const uint32 l_412 = l_46 * uint32(16);
const uint32 l_413 = l_411 + l_412;
const uint32 l_414 = l_47 * uint32(4);
const uint32 l_415 = l_413 + l_414;
const uint32 l_416 = l_415 + uint32(0);
const uint32_4 l_417 = buffer_1[l_416];
const uint32 l_418 = l_416 + uint32(2);
const uint32_4 l_419 = buffer_1[l_418];
const uint32 l_420 = l_98.i_0;
const uint32 l_421 = l_420 & uint32(252645135);
const uint32 l_422 = l_420 >> uint32(4);
const uint32 l_423 = l_422 & uint32(252645135);
const uint32 l_424 = l_417.i_0;
const int8_4 l_425 = reinterpret_cast<int8_4 const&>(l_421);
const int32_4 l_426 = int32_4{
int32(l_425.i_0),int32(l_425.i_1),int32(l_425.i_2),int32(l_425.i_3),};
const int8_4 l_427 = reinterpret_cast<int8_4 const&>(l_424);
const int32_4 l_428 = int32_4{
int32(l_427.i_0),int32(l_427.i_1),int32(l_427.i_2),int32(l_427.i_3),};
#if defined(__gfx1201__)
const int32 l_429 = __builtin_amdgcn_sudot4(true, l_421, true, l_424, 0, false);
#else
const int32 l_429 = l_426.i_0 * l_428.i_0 + l_426.i_1 * l_428.i_1 + l_426.i_2 * l_428.i_2 + l_426.i_3 * l_428.i_3;
#endif
const int32 l_430 = l_arr_90[uint32(2)];
const int32 l_431 = l_430 + l_429;
l_arr_90[uint32(2)] = l_431;
const uint32 l_432 = l_419.i_0;
const int8_4 l_433 = reinterpret_cast<int8_4 const&>(l_423);
const int32_4 l_434 = int32_4{
int32(l_433.i_0),int32(l_433.i_1),int32(l_433.i_2),int32(l_433.i_3),};
const int8_4 l_435 = reinterpret_cast<int8_4 const&>(l_432);
const int32_4 l_436 = int32_4{
int32(l_435.i_0),int32(l_435.i_1),int32(l_435.i_2),int32(l_435.i_3),};
#if defined(__gfx1201__)
const int32 l_437 = __builtin_amdgcn_sudot4(true, l_423, true, l_432, 0, false);
#else
const int32 l_437 = l_434.i_0 * l_436.i_0 + l_434.i_1 * l_436.i_1 + l_434.i_2 * l_436.i_2 + l_434.i_3 * l_436.i_3;
#endif
const int32 l_438 = l_arr_91[uint32(2)];
const int32 l_439 = l_438 + l_437;
l_arr_91[uint32(2)] = l_439;
const uint32 l_440 = l_417.i_0;
const uint32 l_441_bits = uint32(16843009);
const int8_4 l_441 = reinterpret_cast<int8_4 const&>(l_441_bits);
const int32_4 l_442 = int32_4{
int32(l_441.i_0),int32(l_441.i_1),int32(l_441.i_2),int32(l_441.i_3),};
const int8_4 l_443 = reinterpret_cast<int8_4 const&>(l_440);
const int32_4 l_444 = int32_4{
int32(l_443.i_0),int32(l_443.i_1),int32(l_443.i_2),int32(l_443.i_3),};
#if defined(__gfx1201__)
const int32 l_445 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_440, 0, false);
#else
const int32 l_445 = l_442.i_0 * l_444.i_0 + l_442.i_1 * l_444.i_1 + l_442.i_2 * l_444.i_2 + l_442.i_3 * l_444.i_3;
#endif
const int32 l_446 = l_arr_92[uint32(2)];
const int32 l_447 = l_446 + l_445;
l_arr_92[uint32(2)] = l_447;
const uint32 l_448 = l_419.i_0;
const uint32 l_449_bits = uint32(16843009);
const int8_4 l_449 = reinterpret_cast<int8_4 const&>(l_449_bits);
const int32_4 l_450 = int32_4{
int32(l_449.i_0),int32(l_449.i_1),int32(l_449.i_2),int32(l_449.i_3),};
const int8_4 l_451 = reinterpret_cast<int8_4 const&>(l_448);
const int32_4 l_452 = int32_4{
int32(l_451.i_0),int32(l_451.i_1),int32(l_451.i_2),int32(l_451.i_3),};
#if defined(__gfx1201__)
const int32 l_453 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_448, 0, false);
#else
const int32 l_453 = l_450.i_0 * l_452.i_0 + l_450.i_1 * l_452.i_1 + l_450.i_2 * l_452.i_2 + l_450.i_3 * l_452.i_3;
#endif
const int32 l_454 = l_arr_93[uint32(2)];
const int32 l_455 = l_454 + l_453;
l_arr_93[uint32(2)] = l_455;
const uint32 l_456 = l_98.i_1;
const uint32 l_457 = l_456 & uint32(252645135);
const uint32 l_458 = l_456 >> uint32(4);
const uint32 l_459 = l_458 & uint32(252645135);
const uint32 l_460 = l_417.i_1;
const int8_4 l_461 = reinterpret_cast<int8_4 const&>(l_457);
const int32_4 l_462 = int32_4{
int32(l_461.i_0),int32(l_461.i_1),int32(l_461.i_2),int32(l_461.i_3),};
const int8_4 l_463 = reinterpret_cast<int8_4 const&>(l_460);
const int32_4 l_464 = int32_4{
int32(l_463.i_0),int32(l_463.i_1),int32(l_463.i_2),int32(l_463.i_3),};
#if defined(__gfx1201__)
const int32 l_465 = __builtin_amdgcn_sudot4(true, l_457, true, l_460, 0, false);
#else
const int32 l_465 = l_462.i_0 * l_464.i_0 + l_462.i_1 * l_464.i_1 + l_462.i_2 * l_464.i_2 + l_462.i_3 * l_464.i_3;
#endif
const int32 l_466 = l_arr_90[uint32(2)];
const int32 l_467 = l_466 + l_465;
l_arr_90[uint32(2)] = l_467;
const uint32 l_468 = l_419.i_1;
const int8_4 l_469 = reinterpret_cast<int8_4 const&>(l_459);
const int32_4 l_470 = int32_4{
int32(l_469.i_0),int32(l_469.i_1),int32(l_469.i_2),int32(l_469.i_3),};
const int8_4 l_471 = reinterpret_cast<int8_4 const&>(l_468);
const int32_4 l_472 = int32_4{
int32(l_471.i_0),int32(l_471.i_1),int32(l_471.i_2),int32(l_471.i_3),};
#if defined(__gfx1201__)
const int32 l_473 = __builtin_amdgcn_sudot4(true, l_459, true, l_468, 0, false);
#else
const int32 l_473 = l_470.i_0 * l_472.i_0 + l_470.i_1 * l_472.i_1 + l_470.i_2 * l_472.i_2 + l_470.i_3 * l_472.i_3;
#endif
const int32 l_474 = l_arr_91[uint32(2)];
const int32 l_475 = l_474 + l_473;
l_arr_91[uint32(2)] = l_475;
const uint32 l_476 = l_417.i_1;
const uint32 l_477_bits = uint32(16843009);
const int8_4 l_477 = reinterpret_cast<int8_4 const&>(l_477_bits);
const int32_4 l_478 = int32_4{
int32(l_477.i_0),int32(l_477.i_1),int32(l_477.i_2),int32(l_477.i_3),};
const int8_4 l_479 = reinterpret_cast<int8_4 const&>(l_476);
const int32_4 l_480 = int32_4{
int32(l_479.i_0),int32(l_479.i_1),int32(l_479.i_2),int32(l_479.i_3),};
#if defined(__gfx1201__)
const int32 l_481 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_476, 0, false);
#else
const int32 l_481 = l_478.i_0 * l_480.i_0 + l_478.i_1 * l_480.i_1 + l_478.i_2 * l_480.i_2 + l_478.i_3 * l_480.i_3;
#endif
const int32 l_482 = l_arr_92[uint32(2)];
const int32 l_483 = l_482 + l_481;
l_arr_92[uint32(2)] = l_483;
const uint32 l_484 = l_419.i_1;
const uint32 l_485_bits = uint32(16843009);
const int8_4 l_485 = reinterpret_cast<int8_4 const&>(l_485_bits);
const int32_4 l_486 = int32_4{
int32(l_485.i_0),int32(l_485.i_1),int32(l_485.i_2),int32(l_485.i_3),};
const int8_4 l_487 = reinterpret_cast<int8_4 const&>(l_484);
const int32_4 l_488 = int32_4{
int32(l_487.i_0),int32(l_487.i_1),int32(l_487.i_2),int32(l_487.i_3),};
#if defined(__gfx1201__)
const int32 l_489 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_484, 0, false);
#else
const int32 l_489 = l_486.i_0 * l_488.i_0 + l_486.i_1 * l_488.i_1 + l_486.i_2 * l_488.i_2 + l_486.i_3 * l_488.i_3;
#endif
const int32 l_490 = l_arr_93[uint32(2)];
const int32 l_491 = l_490 + l_489;
l_arr_93[uint32(2)] = l_491;
const uint32 l_492 = l_98.i_2;
const uint32 l_493 = l_492 & uint32(252645135);
const uint32 l_494 = l_492 >> uint32(4);
const uint32 l_495 = l_494 & uint32(252645135);
const uint32 l_496 = l_417.i_2;
const int8_4 l_497 = reinterpret_cast<int8_4 const&>(l_493);
const int32_4 l_498 = int32_4{
int32(l_497.i_0),int32(l_497.i_1),int32(l_497.i_2),int32(l_497.i_3),};
const int8_4 l_499 = reinterpret_cast<int8_4 const&>(l_496);
const int32_4 l_500 = int32_4{
int32(l_499.i_0),int32(l_499.i_1),int32(l_499.i_2),int32(l_499.i_3),};
#if defined(__gfx1201__)
const int32 l_501 = __builtin_amdgcn_sudot4(true, l_493, true, l_496, 0, false);
#else
const int32 l_501 = l_498.i_0 * l_500.i_0 + l_498.i_1 * l_500.i_1 + l_498.i_2 * l_500.i_2 + l_498.i_3 * l_500.i_3;
#endif
const int32 l_502 = l_arr_90[uint32(2)];
const int32 l_503 = l_502 + l_501;
l_arr_90[uint32(2)] = l_503;
const uint32 l_504 = l_419.i_2;
const int8_4 l_505 = reinterpret_cast<int8_4 const&>(l_495);
const int32_4 l_506 = int32_4{
int32(l_505.i_0),int32(l_505.i_1),int32(l_505.i_2),int32(l_505.i_3),};
const int8_4 l_507 = reinterpret_cast<int8_4 const&>(l_504);
const int32_4 l_508 = int32_4{
int32(l_507.i_0),int32(l_507.i_1),int32(l_507.i_2),int32(l_507.i_3),};
#if defined(__gfx1201__)
const int32 l_509 = __builtin_amdgcn_sudot4(true, l_495, true, l_504, 0, false);
#else
const int32 l_509 = l_506.i_0 * l_508.i_0 + l_506.i_1 * l_508.i_1 + l_506.i_2 * l_508.i_2 + l_506.i_3 * l_508.i_3;
#endif
const int32 l_510 = l_arr_91[uint32(2)];
const int32 l_511 = l_510 + l_509;
l_arr_91[uint32(2)] = l_511;
const uint32 l_512 = l_417.i_2;
const uint32 l_513_bits = uint32(16843009);
const int8_4 l_513 = reinterpret_cast<int8_4 const&>(l_513_bits);
const int32_4 l_514 = int32_4{
int32(l_513.i_0),int32(l_513.i_1),int32(l_513.i_2),int32(l_513.i_3),};
const int8_4 l_515 = reinterpret_cast<int8_4 const&>(l_512);
const int32_4 l_516 = int32_4{
int32(l_515.i_0),int32(l_515.i_1),int32(l_515.i_2),int32(l_515.i_3),};
#if defined(__gfx1201__)
const int32 l_517 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_512, 0, false);
#else
const int32 l_517 = l_514.i_0 * l_516.i_0 + l_514.i_1 * l_516.i_1 + l_514.i_2 * l_516.i_2 + l_514.i_3 * l_516.i_3;
#endif
const int32 l_518 = l_arr_92[uint32(2)];
const int32 l_519 = l_518 + l_517;
l_arr_92[uint32(2)] = l_519;
const uint32 l_520 = l_419.i_2;
const uint32 l_521_bits = uint32(16843009);
const int8_4 l_521 = reinterpret_cast<int8_4 const&>(l_521_bits);
const int32_4 l_522 = int32_4{
int32(l_521.i_0),int32(l_521.i_1),int32(l_521.i_2),int32(l_521.i_3),};
const int8_4 l_523 = reinterpret_cast<int8_4 const&>(l_520);
const int32_4 l_524 = int32_4{
int32(l_523.i_0),int32(l_523.i_1),int32(l_523.i_2),int32(l_523.i_3),};
#if defined(__gfx1201__)
const int32 l_525 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_520, 0, false);
#else
const int32 l_525 = l_522.i_0 * l_524.i_0 + l_522.i_1 * l_524.i_1 + l_522.i_2 * l_524.i_2 + l_522.i_3 * l_524.i_3;
#endif
const int32 l_526 = l_arr_93[uint32(2)];
const int32 l_527 = l_526 + l_525;
l_arr_93[uint32(2)] = l_527;
const uint32 l_528 = l_98.i_3;
const uint32 l_529 = l_528 & uint32(252645135);
const uint32 l_530 = l_528 >> uint32(4);
const uint32 l_531 = l_530 & uint32(252645135);
const uint32 l_532 = l_417.i_3;
const int8_4 l_533 = reinterpret_cast<int8_4 const&>(l_529);
const int32_4 l_534 = int32_4{
int32(l_533.i_0),int32(l_533.i_1),int32(l_533.i_2),int32(l_533.i_3),};
const int8_4 l_535 = reinterpret_cast<int8_4 const&>(l_532);
const int32_4 l_536 = int32_4{
int32(l_535.i_0),int32(l_535.i_1),int32(l_535.i_2),int32(l_535.i_3),};
#if defined(__gfx1201__)
const int32 l_537 = __builtin_amdgcn_sudot4(true, l_529, true, l_532, 0, false);
#else
const int32 l_537 = l_534.i_0 * l_536.i_0 + l_534.i_1 * l_536.i_1 + l_534.i_2 * l_536.i_2 + l_534.i_3 * l_536.i_3;
#endif
const int32 l_538 = l_arr_90[uint32(2)];
const int32 l_539 = l_538 + l_537;
l_arr_90[uint32(2)] = l_539;
const uint32 l_540 = l_419.i_3;
const int8_4 l_541 = reinterpret_cast<int8_4 const&>(l_531);
const int32_4 l_542 = int32_4{
int32(l_541.i_0),int32(l_541.i_1),int32(l_541.i_2),int32(l_541.i_3),};
const int8_4 l_543 = reinterpret_cast<int8_4 const&>(l_540);
const int32_4 l_544 = int32_4{
int32(l_543.i_0),int32(l_543.i_1),int32(l_543.i_2),int32(l_543.i_3),};
#if defined(__gfx1201__)
const int32 l_545 = __builtin_amdgcn_sudot4(true, l_531, true, l_540, 0, false);
#else
const int32 l_545 = l_542.i_0 * l_544.i_0 + l_542.i_1 * l_544.i_1 + l_542.i_2 * l_544.i_2 + l_542.i_3 * l_544.i_3;
#endif
const int32 l_546 = l_arr_91[uint32(2)];
const int32 l_547 = l_546 + l_545;
l_arr_91[uint32(2)] = l_547;
const uint32 l_548 = l_417.i_3;
const uint32 l_549_bits = uint32(16843009);
const int8_4 l_549 = reinterpret_cast<int8_4 const&>(l_549_bits);
const int32_4 l_550 = int32_4{
int32(l_549.i_0),int32(l_549.i_1),int32(l_549.i_2),int32(l_549.i_3),};
const int8_4 l_551 = reinterpret_cast<int8_4 const&>(l_548);
const int32_4 l_552 = int32_4{
int32(l_551.i_0),int32(l_551.i_1),int32(l_551.i_2),int32(l_551.i_3),};
#if defined(__gfx1201__)
const int32 l_553 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_548, 0, false);
#else
const int32 l_553 = l_550.i_0 * l_552.i_0 + l_550.i_1 * l_552.i_1 + l_550.i_2 * l_552.i_2 + l_550.i_3 * l_552.i_3;
#endif
const int32 l_554 = l_arr_92[uint32(2)];
const int32 l_555 = l_554 + l_553;
l_arr_92[uint32(2)] = l_555;
const uint32 l_556 = l_419.i_3;
const uint32 l_557_bits = uint32(16843009);
const int8_4 l_557 = reinterpret_cast<int8_4 const&>(l_557_bits);
const int32_4 l_558 = int32_4{
int32(l_557.i_0),int32(l_557.i_1),int32(l_557.i_2),int32(l_557.i_3),};
const int8_4 l_559 = reinterpret_cast<int8_4 const&>(l_556);
const int32_4 l_560 = int32_4{
int32(l_559.i_0),int32(l_559.i_1),int32(l_559.i_2),int32(l_559.i_3),};
#if defined(__gfx1201__)
const int32 l_561 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_556, 0, false);
#else
const int32 l_561 = l_558.i_0 * l_560.i_0 + l_558.i_1 * l_560.i_1 + l_558.i_2 * l_560.i_2 + l_558.i_3 * l_560.i_3;
#endif
const int32 l_562 = l_arr_93[uint32(2)];
const int32 l_563 = l_562 + l_561;
l_arr_93[uint32(2)] = l_563;
const uint32 l_564 = l_arr_28[uint32(3)];
const uint32 l_565 = l_1 / uint32(16);
const uint32 l_566 = l_564 * l_565;
const uint32 l_567 = l_46 * uint32(16);
const uint32 l_568 = l_566 + l_567;
const uint32 l_569 = l_47 * uint32(4);
const uint32 l_570 = l_568 + l_569;
const uint32 l_571 = l_570 + uint32(0);
const uint32_4 l_572 = buffer_1[l_571];
const uint32 l_573 = l_571 + uint32(2);
const uint32_4 l_574 = buffer_1[l_573];
const uint32 l_575 = l_98.i_0;
const uint32 l_576 = l_575 & uint32(252645135);
const uint32 l_577 = l_575 >> uint32(4);
const uint32 l_578 = l_577 & uint32(252645135);
const uint32 l_579 = l_572.i_0;
const int8_4 l_580 = reinterpret_cast<int8_4 const&>(l_576);
const int32_4 l_581 = int32_4{
int32(l_580.i_0),int32(l_580.i_1),int32(l_580.i_2),int32(l_580.i_3),};
const int8_4 l_582 = reinterpret_cast<int8_4 const&>(l_579);
const int32_4 l_583 = int32_4{
int32(l_582.i_0),int32(l_582.i_1),int32(l_582.i_2),int32(l_582.i_3),};
#if defined(__gfx1201__)
const int32 l_584 = __builtin_amdgcn_sudot4(true, l_576, true, l_579, 0, false);
#else
const int32 l_584 = l_581.i_0 * l_583.i_0 + l_581.i_1 * l_583.i_1 + l_581.i_2 * l_583.i_2 + l_581.i_3 * l_583.i_3;
#endif
const int32 l_585 = l_arr_90[uint32(3)];
const int32 l_586 = l_585 + l_584;
l_arr_90[uint32(3)] = l_586;
const uint32 l_587 = l_574.i_0;
const int8_4 l_588 = reinterpret_cast<int8_4 const&>(l_578);
const int32_4 l_589 = int32_4{
int32(l_588.i_0),int32(l_588.i_1),int32(l_588.i_2),int32(l_588.i_3),};
const int8_4 l_590 = reinterpret_cast<int8_4 const&>(l_587);
const int32_4 l_591 = int32_4{
int32(l_590.i_0),int32(l_590.i_1),int32(l_590.i_2),int32(l_590.i_3),};
#if defined(__gfx1201__)
const int32 l_592 = __builtin_amdgcn_sudot4(true, l_578, true, l_587, 0, false);
#else
const int32 l_592 = l_589.i_0 * l_591.i_0 + l_589.i_1 * l_591.i_1 + l_589.i_2 * l_591.i_2 + l_589.i_3 * l_591.i_3;
#endif
const int32 l_593 = l_arr_91[uint32(3)];
const int32 l_594 = l_593 + l_592;
l_arr_91[uint32(3)] = l_594;
const uint32 l_595 = l_572.i_0;
const uint32 l_596_bits = uint32(16843009);
const int8_4 l_596 = reinterpret_cast<int8_4 const&>(l_596_bits);
const int32_4 l_597 = int32_4{
int32(l_596.i_0),int32(l_596.i_1),int32(l_596.i_2),int32(l_596.i_3),};
const int8_4 l_598 = reinterpret_cast<int8_4 const&>(l_595);
const int32_4 l_599 = int32_4{
int32(l_598.i_0),int32(l_598.i_1),int32(l_598.i_2),int32(l_598.i_3),};
#if defined(__gfx1201__)
const int32 l_600 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_595, 0, false);
#else
const int32 l_600 = l_597.i_0 * l_599.i_0 + l_597.i_1 * l_599.i_1 + l_597.i_2 * l_599.i_2 + l_597.i_3 * l_599.i_3;
#endif
const int32 l_601 = l_arr_92[uint32(3)];
const int32 l_602 = l_601 + l_600;
l_arr_92[uint32(3)] = l_602;
const uint32 l_603 = l_574.i_0;
const uint32 l_604_bits = uint32(16843009);
const int8_4 l_604 = reinterpret_cast<int8_4 const&>(l_604_bits);
const int32_4 l_605 = int32_4{
int32(l_604.i_0),int32(l_604.i_1),int32(l_604.i_2),int32(l_604.i_3),};
const int8_4 l_606 = reinterpret_cast<int8_4 const&>(l_603);
const int32_4 l_607 = int32_4{
int32(l_606.i_0),int32(l_606.i_1),int32(l_606.i_2),int32(l_606.i_3),};
#if defined(__gfx1201__)
const int32 l_608 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_603, 0, false);
#else
const int32 l_608 = l_605.i_0 * l_607.i_0 + l_605.i_1 * l_607.i_1 + l_605.i_2 * l_607.i_2 + l_605.i_3 * l_607.i_3;
#endif
const int32 l_609 = l_arr_93[uint32(3)];
const int32 l_610 = l_609 + l_608;
l_arr_93[uint32(3)] = l_610;
const uint32 l_611 = l_98.i_1;
const uint32 l_612 = l_611 & uint32(252645135);
const uint32 l_613 = l_611 >> uint32(4);
const uint32 l_614 = l_613 & uint32(252645135);
const uint32 l_615 = l_572.i_1;
const int8_4 l_616 = reinterpret_cast<int8_4 const&>(l_612);
const int32_4 l_617 = int32_4{
int32(l_616.i_0),int32(l_616.i_1),int32(l_616.i_2),int32(l_616.i_3),};
const int8_4 l_618 = reinterpret_cast<int8_4 const&>(l_615);
const int32_4 l_619 = int32_4{
int32(l_618.i_0),int32(l_618.i_1),int32(l_618.i_2),int32(l_618.i_3),};
#if defined(__gfx1201__)
const int32 l_620 = __builtin_amdgcn_sudot4(true, l_612, true, l_615, 0, false);
#else
const int32 l_620 = l_617.i_0 * l_619.i_0 + l_617.i_1 * l_619.i_1 + l_617.i_2 * l_619.i_2 + l_617.i_3 * l_619.i_3;
#endif
const int32 l_621 = l_arr_90[uint32(3)];
const int32 l_622 = l_621 + l_620;
l_arr_90[uint32(3)] = l_622;
const uint32 l_623 = l_574.i_1;
const int8_4 l_624 = reinterpret_cast<int8_4 const&>(l_614);
const int32_4 l_625 = int32_4{
int32(l_624.i_0),int32(l_624.i_1),int32(l_624.i_2),int32(l_624.i_3),};
const int8_4 l_626 = reinterpret_cast<int8_4 const&>(l_623);
const int32_4 l_627 = int32_4{
int32(l_626.i_0),int32(l_626.i_1),int32(l_626.i_2),int32(l_626.i_3),};
#if defined(__gfx1201__)
const int32 l_628 = __builtin_amdgcn_sudot4(true, l_614, true, l_623, 0, false);
#else
const int32 l_628 = l_625.i_0 * l_627.i_0 + l_625.i_1 * l_627.i_1 + l_625.i_2 * l_627.i_2 + l_625.i_3 * l_627.i_3;
#endif
const int32 l_629 = l_arr_91[uint32(3)];
const int32 l_630 = l_629 + l_628;
l_arr_91[uint32(3)] = l_630;
const uint32 l_631 = l_572.i_1;
const uint32 l_632_bits = uint32(16843009);
const int8_4 l_632 = reinterpret_cast<int8_4 const&>(l_632_bits);
const int32_4 l_633 = int32_4{
int32(l_632.i_0),int32(l_632.i_1),int32(l_632.i_2),int32(l_632.i_3),};
const int8_4 l_634 = reinterpret_cast<int8_4 const&>(l_631);
const int32_4 l_635 = int32_4{
int32(l_634.i_0),int32(l_634.i_1),int32(l_634.i_2),int32(l_634.i_3),};
#if defined(__gfx1201__)
const int32 l_636 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_631, 0, false);
#else
const int32 l_636 = l_633.i_0 * l_635.i_0 + l_633.i_1 * l_635.i_1 + l_633.i_2 * l_635.i_2 + l_633.i_3 * l_635.i_3;
#endif
const int32 l_637 = l_arr_92[uint32(3)];
const int32 l_638 = l_637 + l_636;
l_arr_92[uint32(3)] = l_638;
const uint32 l_639 = l_574.i_1;
const uint32 l_640_bits = uint32(16843009);
const int8_4 l_640 = reinterpret_cast<int8_4 const&>(l_640_bits);
const int32_4 l_641 = int32_4{
int32(l_640.i_0),int32(l_640.i_1),int32(l_640.i_2),int32(l_640.i_3),};
const int8_4 l_642 = reinterpret_cast<int8_4 const&>(l_639);
const int32_4 l_643 = int32_4{
int32(l_642.i_0),int32(l_642.i_1),int32(l_642.i_2),int32(l_642.i_3),};
#if defined(__gfx1201__)
const int32 l_644 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_639, 0, false);
#else
const int32 l_644 = l_641.i_0 * l_643.i_0 + l_641.i_1 * l_643.i_1 + l_641.i_2 * l_643.i_2 + l_641.i_3 * l_643.i_3;
#endif
const int32 l_645 = l_arr_93[uint32(3)];
const int32 l_646 = l_645 + l_644;
l_arr_93[uint32(3)] = l_646;
const uint32 l_647 = l_98.i_2;
const uint32 l_648 = l_647 & uint32(252645135);
const uint32 l_649 = l_647 >> uint32(4);
const uint32 l_650 = l_649 & uint32(252645135);
const uint32 l_651 = l_572.i_2;
const int8_4 l_652 = reinterpret_cast<int8_4 const&>(l_648);
const int32_4 l_653 = int32_4{
int32(l_652.i_0),int32(l_652.i_1),int32(l_652.i_2),int32(l_652.i_3),};
const int8_4 l_654 = reinterpret_cast<int8_4 const&>(l_651);
const int32_4 l_655 = int32_4{
int32(l_654.i_0),int32(l_654.i_1),int32(l_654.i_2),int32(l_654.i_3),};
#if defined(__gfx1201__)
const int32 l_656 = __builtin_amdgcn_sudot4(true, l_648, true, l_651, 0, false);
#else
const int32 l_656 = l_653.i_0 * l_655.i_0 + l_653.i_1 * l_655.i_1 + l_653.i_2 * l_655.i_2 + l_653.i_3 * l_655.i_3;
#endif
const int32 l_657 = l_arr_90[uint32(3)];
const int32 l_658 = l_657 + l_656;
l_arr_90[uint32(3)] = l_658;
const uint32 l_659 = l_574.i_2;
const int8_4 l_660 = reinterpret_cast<int8_4 const&>(l_650);
const int32_4 l_661 = int32_4{
int32(l_660.i_0),int32(l_660.i_1),int32(l_660.i_2),int32(l_660.i_3),};
const int8_4 l_662 = reinterpret_cast<int8_4 const&>(l_659);
const int32_4 l_663 = int32_4{
int32(l_662.i_0),int32(l_662.i_1),int32(l_662.i_2),int32(l_662.i_3),};
#if defined(__gfx1201__)
const int32 l_664 = __builtin_amdgcn_sudot4(true, l_650, true, l_659, 0, false);
#else
const int32 l_664 = l_661.i_0 * l_663.i_0 + l_661.i_1 * l_663.i_1 + l_661.i_2 * l_663.i_2 + l_661.i_3 * l_663.i_3;
#endif
const int32 l_665 = l_arr_91[uint32(3)];
const int32 l_666 = l_665 + l_664;
l_arr_91[uint32(3)] = l_666;
const uint32 l_667 = l_572.i_2;
const uint32 l_668_bits = uint32(16843009);
const int8_4 l_668 = reinterpret_cast<int8_4 const&>(l_668_bits);
const int32_4 l_669 = int32_4{
int32(l_668.i_0),int32(l_668.i_1),int32(l_668.i_2),int32(l_668.i_3),};
const int8_4 l_670 = reinterpret_cast<int8_4 const&>(l_667);
const int32_4 l_671 = int32_4{
int32(l_670.i_0),int32(l_670.i_1),int32(l_670.i_2),int32(l_670.i_3),};
#if defined(__gfx1201__)
const int32 l_672 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_667, 0, false);
#else
const int32 l_672 = l_669.i_0 * l_671.i_0 + l_669.i_1 * l_671.i_1 + l_669.i_2 * l_671.i_2 + l_669.i_3 * l_671.i_3;
#endif
const int32 l_673 = l_arr_92[uint32(3)];
const int32 l_674 = l_673 + l_672;
l_arr_92[uint32(3)] = l_674;
const uint32 l_675 = l_574.i_2;
const uint32 l_676_bits = uint32(16843009);
const int8_4 l_676 = reinterpret_cast<int8_4 const&>(l_676_bits);
const int32_4 l_677 = int32_4{
int32(l_676.i_0),int32(l_676.i_1),int32(l_676.i_2),int32(l_676.i_3),};
const int8_4 l_678 = reinterpret_cast<int8_4 const&>(l_675);
const int32_4 l_679 = int32_4{
int32(l_678.i_0),int32(l_678.i_1),int32(l_678.i_2),int32(l_678.i_3),};
#if defined(__gfx1201__)
const int32 l_680 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_675, 0, false);
#else
const int32 l_680 = l_677.i_0 * l_679.i_0 + l_677.i_1 * l_679.i_1 + l_677.i_2 * l_679.i_2 + l_677.i_3 * l_679.i_3;
#endif
const int32 l_681 = l_arr_93[uint32(3)];
const int32 l_682 = l_681 + l_680;
l_arr_93[uint32(3)] = l_682;
const uint32 l_683 = l_98.i_3;
const uint32 l_684 = l_683 & uint32(252645135);
const uint32 l_685 = l_683 >> uint32(4);
const uint32 l_686 = l_685 & uint32(252645135);
const uint32 l_687 = l_572.i_3;
const int8_4 l_688 = reinterpret_cast<int8_4 const&>(l_684);
const int32_4 l_689 = int32_4{
int32(l_688.i_0),int32(l_688.i_1),int32(l_688.i_2),int32(l_688.i_3),};
const int8_4 l_690 = reinterpret_cast<int8_4 const&>(l_687);
const int32_4 l_691 = int32_4{
int32(l_690.i_0),int32(l_690.i_1),int32(l_690.i_2),int32(l_690.i_3),};
#if defined(__gfx1201__)
const int32 l_692 = __builtin_amdgcn_sudot4(true, l_684, true, l_687, 0, false);
#else
const int32 l_692 = l_689.i_0 * l_691.i_0 + l_689.i_1 * l_691.i_1 + l_689.i_2 * l_691.i_2 + l_689.i_3 * l_691.i_3;
#endif
const int32 l_693 = l_arr_90[uint32(3)];
const int32 l_694 = l_693 + l_692;
l_arr_90[uint32(3)] = l_694;
const uint32 l_695 = l_574.i_3;
const int8_4 l_696 = reinterpret_cast<int8_4 const&>(l_686);
const int32_4 l_697 = int32_4{
int32(l_696.i_0),int32(l_696.i_1),int32(l_696.i_2),int32(l_696.i_3),};
const int8_4 l_698 = reinterpret_cast<int8_4 const&>(l_695);
const int32_4 l_699 = int32_4{
int32(l_698.i_0),int32(l_698.i_1),int32(l_698.i_2),int32(l_698.i_3),};
#if defined(__gfx1201__)
const int32 l_700 = __builtin_amdgcn_sudot4(true, l_686, true, l_695, 0, false);
#else
const int32 l_700 = l_697.i_0 * l_699.i_0 + l_697.i_1 * l_699.i_1 + l_697.i_2 * l_699.i_2 + l_697.i_3 * l_699.i_3;
#endif
const int32 l_701 = l_arr_91[uint32(3)];
const int32 l_702 = l_701 + l_700;
l_arr_91[uint32(3)] = l_702;
const uint32 l_703 = l_572.i_3;
const uint32 l_704_bits = uint32(16843009);
const int8_4 l_704 = reinterpret_cast<int8_4 const&>(l_704_bits);
const int32_4 l_705 = int32_4{
int32(l_704.i_0),int32(l_704.i_1),int32(l_704.i_2),int32(l_704.i_3),};
const int8_4 l_706 = reinterpret_cast<int8_4 const&>(l_703);
const int32_4 l_707 = int32_4{
int32(l_706.i_0),int32(l_706.i_1),int32(l_706.i_2),int32(l_706.i_3),};
#if defined(__gfx1201__)
const int32 l_708 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_703, 0, false);
#else
const int32 l_708 = l_705.i_0 * l_707.i_0 + l_705.i_1 * l_707.i_1 + l_705.i_2 * l_707.i_2 + l_705.i_3 * l_707.i_3;
#endif
const int32 l_709 = l_arr_92[uint32(3)];
const int32 l_710 = l_709 + l_708;
l_arr_92[uint32(3)] = l_710;
const uint32 l_711 = l_574.i_3;
const uint32 l_712_bits = uint32(16843009);
const int8_4 l_712 = reinterpret_cast<int8_4 const&>(l_712_bits);
const int32_4 l_713 = int32_4{
int32(l_712.i_0),int32(l_712.i_1),int32(l_712.i_2),int32(l_712.i_3),};
const int8_4 l_714 = reinterpret_cast<int8_4 const&>(l_711);
const int32_4 l_715 = int32_4{
int32(l_714.i_0),int32(l_714.i_1),int32(l_714.i_2),int32(l_714.i_3),};
#if defined(__gfx1201__)
const int32 l_716 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_711, 0, false);
#else
const int32 l_716 = l_713.i_0 * l_715.i_0 + l_713.i_1 * l_715.i_1 + l_713.i_2 * l_715.i_2 + l_713.i_3 * l_715.i_3;
#endif
const int32 l_717 = l_arr_93[uint32(3)];
const int32 l_718 = l_717 + l_716;
l_arr_93[uint32(3)] = l_718;
const uint32 l_719 = l_50 + uint32(1);
const uint32 l_720 = l_47 * uint32(2);
const uint32 l_721 = l_719 + l_720;
const uint32 l_722 = l_721 + uint32(1);
const uint32_4 l_723 = buffer_0[l_722];
const uint32 l_724 = l_arr_28[uint32(0)];
const uint32 l_725 = l_1 / uint32(16);
const uint32 l_726 = l_724 * l_725;
const uint32 l_727 = l_46 * uint32(16);
const uint32 l_728 = l_726 + l_727;
const uint32 l_729 = l_47 * uint32(4);
const uint32 l_730 = l_728 + l_729;
const uint32 l_731 = l_730 + uint32(1);
const uint32_4 l_732 = buffer_1[l_731];
const uint32 l_733 = l_731 + uint32(2);
const uint32_4 l_734 = buffer_1[l_733];
const uint32 l_735 = l_723.i_0;
const uint32 l_736 = l_735 & uint32(252645135);
const uint32 l_737 = l_735 >> uint32(4);
const uint32 l_738 = l_737 & uint32(252645135);
const uint32 l_739 = l_732.i_0;
const int8_4 l_740 = reinterpret_cast<int8_4 const&>(l_736);
const int32_4 l_741 = int32_4{
int32(l_740.i_0),int32(l_740.i_1),int32(l_740.i_2),int32(l_740.i_3),};
const int8_4 l_742 = reinterpret_cast<int8_4 const&>(l_739);
const int32_4 l_743 = int32_4{
int32(l_742.i_0),int32(l_742.i_1),int32(l_742.i_2),int32(l_742.i_3),};
#if defined(__gfx1201__)
const int32 l_744 = __builtin_amdgcn_sudot4(true, l_736, true, l_739, 0, false);
#else
const int32 l_744 = l_741.i_0 * l_743.i_0 + l_741.i_1 * l_743.i_1 + l_741.i_2 * l_743.i_2 + l_741.i_3 * l_743.i_3;
#endif
const int32 l_745 = l_arr_90[uint32(0)];
const int32 l_746 = l_745 + l_744;
l_arr_90[uint32(0)] = l_746;
const uint32 l_747 = l_734.i_0;
const int8_4 l_748 = reinterpret_cast<int8_4 const&>(l_738);
const int32_4 l_749 = int32_4{
int32(l_748.i_0),int32(l_748.i_1),int32(l_748.i_2),int32(l_748.i_3),};
const int8_4 l_750 = reinterpret_cast<int8_4 const&>(l_747);
const int32_4 l_751 = int32_4{
int32(l_750.i_0),int32(l_750.i_1),int32(l_750.i_2),int32(l_750.i_3),};
#if defined(__gfx1201__)
const int32 l_752 = __builtin_amdgcn_sudot4(true, l_738, true, l_747, 0, false);
#else
const int32 l_752 = l_749.i_0 * l_751.i_0 + l_749.i_1 * l_751.i_1 + l_749.i_2 * l_751.i_2 + l_749.i_3 * l_751.i_3;
#endif
const int32 l_753 = l_arr_91[uint32(0)];
const int32 l_754 = l_753 + l_752;
l_arr_91[uint32(0)] = l_754;
const uint32 l_755 = l_732.i_0;
const uint32 l_756_bits = uint32(16843009);
const int8_4 l_756 = reinterpret_cast<int8_4 const&>(l_756_bits);
const int32_4 l_757 = int32_4{
int32(l_756.i_0),int32(l_756.i_1),int32(l_756.i_2),int32(l_756.i_3),};
const int8_4 l_758 = reinterpret_cast<int8_4 const&>(l_755);
const int32_4 l_759 = int32_4{
int32(l_758.i_0),int32(l_758.i_1),int32(l_758.i_2),int32(l_758.i_3),};
#if defined(__gfx1201__)
const int32 l_760 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_755, 0, false);
#else
const int32 l_760 = l_757.i_0 * l_759.i_0 + l_757.i_1 * l_759.i_1 + l_757.i_2 * l_759.i_2 + l_757.i_3 * l_759.i_3;
#endif
const int32 l_761 = l_arr_92[uint32(0)];
const int32 l_762 = l_761 + l_760;
l_arr_92[uint32(0)] = l_762;
const uint32 l_763 = l_734.i_0;
const uint32 l_764_bits = uint32(16843009);
const int8_4 l_764 = reinterpret_cast<int8_4 const&>(l_764_bits);
const int32_4 l_765 = int32_4{
int32(l_764.i_0),int32(l_764.i_1),int32(l_764.i_2),int32(l_764.i_3),};
const int8_4 l_766 = reinterpret_cast<int8_4 const&>(l_763);
const int32_4 l_767 = int32_4{
int32(l_766.i_0),int32(l_766.i_1),int32(l_766.i_2),int32(l_766.i_3),};
#if defined(__gfx1201__)
const int32 l_768 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_763, 0, false);
#else
const int32 l_768 = l_765.i_0 * l_767.i_0 + l_765.i_1 * l_767.i_1 + l_765.i_2 * l_767.i_2 + l_765.i_3 * l_767.i_3;
#endif
const int32 l_769 = l_arr_93[uint32(0)];
const int32 l_770 = l_769 + l_768;
l_arr_93[uint32(0)] = l_770;
const uint32 l_771 = l_723.i_1;
const uint32 l_772 = l_771 & uint32(252645135);
const uint32 l_773 = l_771 >> uint32(4);
const uint32 l_774 = l_773 & uint32(252645135);
const uint32 l_775 = l_732.i_1;
const int8_4 l_776 = reinterpret_cast<int8_4 const&>(l_772);
const int32_4 l_777 = int32_4{
int32(l_776.i_0),int32(l_776.i_1),int32(l_776.i_2),int32(l_776.i_3),};
const int8_4 l_778 = reinterpret_cast<int8_4 const&>(l_775);
const int32_4 l_779 = int32_4{
int32(l_778.i_0),int32(l_778.i_1),int32(l_778.i_2),int32(l_778.i_3),};
#if defined(__gfx1201__)
const int32 l_780 = __builtin_amdgcn_sudot4(true, l_772, true, l_775, 0, false);
#else
const int32 l_780 = l_777.i_0 * l_779.i_0 + l_777.i_1 * l_779.i_1 + l_777.i_2 * l_779.i_2 + l_777.i_3 * l_779.i_3;
#endif
const int32 l_781 = l_arr_90[uint32(0)];
const int32 l_782 = l_781 + l_780;
l_arr_90[uint32(0)] = l_782;
const uint32 l_783 = l_734.i_1;
const int8_4 l_784 = reinterpret_cast<int8_4 const&>(l_774);
const int32_4 l_785 = int32_4{
int32(l_784.i_0),int32(l_784.i_1),int32(l_784.i_2),int32(l_784.i_3),};
const int8_4 l_786 = reinterpret_cast<int8_4 const&>(l_783);
const int32_4 l_787 = int32_4{
int32(l_786.i_0),int32(l_786.i_1),int32(l_786.i_2),int32(l_786.i_3),};
#if defined(__gfx1201__)
const int32 l_788 = __builtin_amdgcn_sudot4(true, l_774, true, l_783, 0, false);
#else
const int32 l_788 = l_785.i_0 * l_787.i_0 + l_785.i_1 * l_787.i_1 + l_785.i_2 * l_787.i_2 + l_785.i_3 * l_787.i_3;
#endif
const int32 l_789 = l_arr_91[uint32(0)];
const int32 l_790 = l_789 + l_788;
l_arr_91[uint32(0)] = l_790;
const uint32 l_791 = l_732.i_1;
const uint32 l_792_bits = uint32(16843009);
const int8_4 l_792 = reinterpret_cast<int8_4 const&>(l_792_bits);
const int32_4 l_793 = int32_4{
int32(l_792.i_0),int32(l_792.i_1),int32(l_792.i_2),int32(l_792.i_3),};
const int8_4 l_794 = reinterpret_cast<int8_4 const&>(l_791);
const int32_4 l_795 = int32_4{
int32(l_794.i_0),int32(l_794.i_1),int32(l_794.i_2),int32(l_794.i_3),};
#if defined(__gfx1201__)
const int32 l_796 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_791, 0, false);
#else
const int32 l_796 = l_793.i_0 * l_795.i_0 + l_793.i_1 * l_795.i_1 + l_793.i_2 * l_795.i_2 + l_793.i_3 * l_795.i_3;
#endif
const int32 l_797 = l_arr_92[uint32(0)];
const int32 l_798 = l_797 + l_796;
l_arr_92[uint32(0)] = l_798;
const uint32 l_799 = l_734.i_1;
const uint32 l_800_bits = uint32(16843009);
const int8_4 l_800 = reinterpret_cast<int8_4 const&>(l_800_bits);
const int32_4 l_801 = int32_4{
int32(l_800.i_0),int32(l_800.i_1),int32(l_800.i_2),int32(l_800.i_3),};
const int8_4 l_802 = reinterpret_cast<int8_4 const&>(l_799);
const int32_4 l_803 = int32_4{
int32(l_802.i_0),int32(l_802.i_1),int32(l_802.i_2),int32(l_802.i_3),};
#if defined(__gfx1201__)
const int32 l_804 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_799, 0, false);
#else
const int32 l_804 = l_801.i_0 * l_803.i_0 + l_801.i_1 * l_803.i_1 + l_801.i_2 * l_803.i_2 + l_801.i_3 * l_803.i_3;
#endif
const int32 l_805 = l_arr_93[uint32(0)];
const int32 l_806 = l_805 + l_804;
l_arr_93[uint32(0)] = l_806;
const uint32 l_807 = l_723.i_2;
const uint32 l_808 = l_807 & uint32(252645135);
const uint32 l_809 = l_807 >> uint32(4);
const uint32 l_810 = l_809 & uint32(252645135);
const uint32 l_811 = l_732.i_2;
const int8_4 l_812 = reinterpret_cast<int8_4 const&>(l_808);
const int32_4 l_813 = int32_4{
int32(l_812.i_0),int32(l_812.i_1),int32(l_812.i_2),int32(l_812.i_3),};
const int8_4 l_814 = reinterpret_cast<int8_4 const&>(l_811);
const int32_4 l_815 = int32_4{
int32(l_814.i_0),int32(l_814.i_1),int32(l_814.i_2),int32(l_814.i_3),};
#if defined(__gfx1201__)
const int32 l_816 = __builtin_amdgcn_sudot4(true, l_808, true, l_811, 0, false);
#else
const int32 l_816 = l_813.i_0 * l_815.i_0 + l_813.i_1 * l_815.i_1 + l_813.i_2 * l_815.i_2 + l_813.i_3 * l_815.i_3;
#endif
const int32 l_817 = l_arr_90[uint32(0)];
const int32 l_818 = l_817 + l_816;
l_arr_90[uint32(0)] = l_818;
const uint32 l_819 = l_734.i_2;
const int8_4 l_820 = reinterpret_cast<int8_4 const&>(l_810);
const int32_4 l_821 = int32_4{
int32(l_820.i_0),int32(l_820.i_1),int32(l_820.i_2),int32(l_820.i_3),};
const int8_4 l_822 = reinterpret_cast<int8_4 const&>(l_819);
const int32_4 l_823 = int32_4{
int32(l_822.i_0),int32(l_822.i_1),int32(l_822.i_2),int32(l_822.i_3),};
#if defined(__gfx1201__)
const int32 l_824 = __builtin_amdgcn_sudot4(true, l_810, true, l_819, 0, false);
#else
const int32 l_824 = l_821.i_0 * l_823.i_0 + l_821.i_1 * l_823.i_1 + l_821.i_2 * l_823.i_2 + l_821.i_3 * l_823.i_3;
#endif
const int32 l_825 = l_arr_91[uint32(0)];
const int32 l_826 = l_825 + l_824;
l_arr_91[uint32(0)] = l_826;
const uint32 l_827 = l_732.i_2;
const uint32 l_828_bits = uint32(16843009);
const int8_4 l_828 = reinterpret_cast<int8_4 const&>(l_828_bits);
const int32_4 l_829 = int32_4{
int32(l_828.i_0),int32(l_828.i_1),int32(l_828.i_2),int32(l_828.i_3),};
const int8_4 l_830 = reinterpret_cast<int8_4 const&>(l_827);
const int32_4 l_831 = int32_4{
int32(l_830.i_0),int32(l_830.i_1),int32(l_830.i_2),int32(l_830.i_3),};
#if defined(__gfx1201__)
const int32 l_832 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_827, 0, false);
#else
const int32 l_832 = l_829.i_0 * l_831.i_0 + l_829.i_1 * l_831.i_1 + l_829.i_2 * l_831.i_2 + l_829.i_3 * l_831.i_3;
#endif
const int32 l_833 = l_arr_92[uint32(0)];
const int32 l_834 = l_833 + l_832;
l_arr_92[uint32(0)] = l_834;
const uint32 l_835 = l_734.i_2;
const uint32 l_836_bits = uint32(16843009);
const int8_4 l_836 = reinterpret_cast<int8_4 const&>(l_836_bits);
const int32_4 l_837 = int32_4{
int32(l_836.i_0),int32(l_836.i_1),int32(l_836.i_2),int32(l_836.i_3),};
const int8_4 l_838 = reinterpret_cast<int8_4 const&>(l_835);
const int32_4 l_839 = int32_4{
int32(l_838.i_0),int32(l_838.i_1),int32(l_838.i_2),int32(l_838.i_3),};
#if defined(__gfx1201__)
const int32 l_840 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_835, 0, false);
#else
const int32 l_840 = l_837.i_0 * l_839.i_0 + l_837.i_1 * l_839.i_1 + l_837.i_2 * l_839.i_2 + l_837.i_3 * l_839.i_3;
#endif
const int32 l_841 = l_arr_93[uint32(0)];
const int32 l_842 = l_841 + l_840;
l_arr_93[uint32(0)] = l_842;
const uint32 l_843 = l_723.i_3;
const uint32 l_844 = l_843 & uint32(252645135);
const uint32 l_845 = l_843 >> uint32(4);
const uint32 l_846 = l_845 & uint32(252645135);
const uint32 l_847 = l_732.i_3;
const int8_4 l_848 = reinterpret_cast<int8_4 const&>(l_844);
const int32_4 l_849 = int32_4{
int32(l_848.i_0),int32(l_848.i_1),int32(l_848.i_2),int32(l_848.i_3),};
const int8_4 l_850 = reinterpret_cast<int8_4 const&>(l_847);
const int32_4 l_851 = int32_4{
int32(l_850.i_0),int32(l_850.i_1),int32(l_850.i_2),int32(l_850.i_3),};
#if defined(__gfx1201__)
const int32 l_852 = __builtin_amdgcn_sudot4(true, l_844, true, l_847, 0, false);
#else
const int32 l_852 = l_849.i_0 * l_851.i_0 + l_849.i_1 * l_851.i_1 + l_849.i_2 * l_851.i_2 + l_849.i_3 * l_851.i_3;
#endif
const int32 l_853 = l_arr_90[uint32(0)];
const int32 l_854 = l_853 + l_852;
l_arr_90[uint32(0)] = l_854;
const uint32 l_855 = l_734.i_3;
const int8_4 l_856 = reinterpret_cast<int8_4 const&>(l_846);
const int32_4 l_857 = int32_4{
int32(l_856.i_0),int32(l_856.i_1),int32(l_856.i_2),int32(l_856.i_3),};
const int8_4 l_858 = reinterpret_cast<int8_4 const&>(l_855);
const int32_4 l_859 = int32_4{
int32(l_858.i_0),int32(l_858.i_1),int32(l_858.i_2),int32(l_858.i_3),};
#if defined(__gfx1201__)
const int32 l_860 = __builtin_amdgcn_sudot4(true, l_846, true, l_855, 0, false);
#else
const int32 l_860 = l_857.i_0 * l_859.i_0 + l_857.i_1 * l_859.i_1 + l_857.i_2 * l_859.i_2 + l_857.i_3 * l_859.i_3;
#endif
const int32 l_861 = l_arr_91[uint32(0)];
const int32 l_862 = l_861 + l_860;
l_arr_91[uint32(0)] = l_862;
const uint32 l_863 = l_732.i_3;
const uint32 l_864_bits = uint32(16843009);
const int8_4 l_864 = reinterpret_cast<int8_4 const&>(l_864_bits);
const int32_4 l_865 = int32_4{
int32(l_864.i_0),int32(l_864.i_1),int32(l_864.i_2),int32(l_864.i_3),};
const int8_4 l_866 = reinterpret_cast<int8_4 const&>(l_863);
const int32_4 l_867 = int32_4{
int32(l_866.i_0),int32(l_866.i_1),int32(l_866.i_2),int32(l_866.i_3),};
#if defined(__gfx1201__)
const int32 l_868 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_863, 0, false);
#else
const int32 l_868 = l_865.i_0 * l_867.i_0 + l_865.i_1 * l_867.i_1 + l_865.i_2 * l_867.i_2 + l_865.i_3 * l_867.i_3;
#endif
const int32 l_869 = l_arr_92[uint32(0)];
const int32 l_870 = l_869 + l_868;
l_arr_92[uint32(0)] = l_870;
const uint32 l_871 = l_734.i_3;
const uint32 l_872_bits = uint32(16843009);
const int8_4 l_872 = reinterpret_cast<int8_4 const&>(l_872_bits);
const int32_4 l_873 = int32_4{
int32(l_872.i_0),int32(l_872.i_1),int32(l_872.i_2),int32(l_872.i_3),};
const int8_4 l_874 = reinterpret_cast<int8_4 const&>(l_871);
const int32_4 l_875 = int32_4{
int32(l_874.i_0),int32(l_874.i_1),int32(l_874.i_2),int32(l_874.i_3),};
#if defined(__gfx1201__)
const int32 l_876 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_871, 0, false);
#else
const int32 l_876 = l_873.i_0 * l_875.i_0 + l_873.i_1 * l_875.i_1 + l_873.i_2 * l_875.i_2 + l_873.i_3 * l_875.i_3;
#endif
const int32 l_877 = l_arr_93[uint32(0)];
const int32 l_878 = l_877 + l_876;
l_arr_93[uint32(0)] = l_878;
const uint32 l_879 = l_arr_28[uint32(1)];
const uint32 l_880 = l_1 / uint32(16);
const uint32 l_881 = l_879 * l_880;
const uint32 l_882 = l_46 * uint32(16);
const uint32 l_883 = l_881 + l_882;
const uint32 l_884 = l_47 * uint32(4);
const uint32 l_885 = l_883 + l_884;
const uint32 l_886 = l_885 + uint32(1);
const uint32_4 l_887 = buffer_1[l_886];
const uint32 l_888 = l_886 + uint32(2);
const uint32_4 l_889 = buffer_1[l_888];
const uint32 l_890 = l_723.i_0;
const uint32 l_891 = l_890 & uint32(252645135);
const uint32 l_892 = l_890 >> uint32(4);
const uint32 l_893 = l_892 & uint32(252645135);
const uint32 l_894 = l_887.i_0;
const int8_4 l_895 = reinterpret_cast<int8_4 const&>(l_891);
const int32_4 l_896 = int32_4{
int32(l_895.i_0),int32(l_895.i_1),int32(l_895.i_2),int32(l_895.i_3),};
const int8_4 l_897 = reinterpret_cast<int8_4 const&>(l_894);
const int32_4 l_898 = int32_4{
int32(l_897.i_0),int32(l_897.i_1),int32(l_897.i_2),int32(l_897.i_3),};
#if defined(__gfx1201__)
const int32 l_899 = __builtin_amdgcn_sudot4(true, l_891, true, l_894, 0, false);
#else
const int32 l_899 = l_896.i_0 * l_898.i_0 + l_896.i_1 * l_898.i_1 + l_896.i_2 * l_898.i_2 + l_896.i_3 * l_898.i_3;
#endif
const int32 l_900 = l_arr_90[uint32(1)];
const int32 l_901 = l_900 + l_899;
l_arr_90[uint32(1)] = l_901;
const uint32 l_902 = l_889.i_0;
const int8_4 l_903 = reinterpret_cast<int8_4 const&>(l_893);
const int32_4 l_904 = int32_4{
int32(l_903.i_0),int32(l_903.i_1),int32(l_903.i_2),int32(l_903.i_3),};
const int8_4 l_905 = reinterpret_cast<int8_4 const&>(l_902);
const int32_4 l_906 = int32_4{
int32(l_905.i_0),int32(l_905.i_1),int32(l_905.i_2),int32(l_905.i_3),};
#if defined(__gfx1201__)
const int32 l_907 = __builtin_amdgcn_sudot4(true, l_893, true, l_902, 0, false);
#else
const int32 l_907 = l_904.i_0 * l_906.i_0 + l_904.i_1 * l_906.i_1 + l_904.i_2 * l_906.i_2 + l_904.i_3 * l_906.i_3;
#endif
const int32 l_908 = l_arr_91[uint32(1)];
const int32 l_909 = l_908 + l_907;
l_arr_91[uint32(1)] = l_909;
const uint32 l_910 = l_887.i_0;
const uint32 l_911_bits = uint32(16843009);
const int8_4 l_911 = reinterpret_cast<int8_4 const&>(l_911_bits);
const int32_4 l_912 = int32_4{
int32(l_911.i_0),int32(l_911.i_1),int32(l_911.i_2),int32(l_911.i_3),};
const int8_4 l_913 = reinterpret_cast<int8_4 const&>(l_910);
const int32_4 l_914 = int32_4{
int32(l_913.i_0),int32(l_913.i_1),int32(l_913.i_2),int32(l_913.i_3),};
#if defined(__gfx1201__)
const int32 l_915 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_910, 0, false);
#else
const int32 l_915 = l_912.i_0 * l_914.i_0 + l_912.i_1 * l_914.i_1 + l_912.i_2 * l_914.i_2 + l_912.i_3 * l_914.i_3;
#endif
const int32 l_916 = l_arr_92[uint32(1)];
const int32 l_917 = l_916 + l_915;
l_arr_92[uint32(1)] = l_917;
const uint32 l_918 = l_889.i_0;
const uint32 l_919_bits = uint32(16843009);
const int8_4 l_919 = reinterpret_cast<int8_4 const&>(l_919_bits);
const int32_4 l_920 = int32_4{
int32(l_919.i_0),int32(l_919.i_1),int32(l_919.i_2),int32(l_919.i_3),};
const int8_4 l_921 = reinterpret_cast<int8_4 const&>(l_918);
const int32_4 l_922 = int32_4{
int32(l_921.i_0),int32(l_921.i_1),int32(l_921.i_2),int32(l_921.i_3),};
#if defined(__gfx1201__)
const int32 l_923 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_918, 0, false);
#else
const int32 l_923 = l_920.i_0 * l_922.i_0 + l_920.i_1 * l_922.i_1 + l_920.i_2 * l_922.i_2 + l_920.i_3 * l_922.i_3;
#endif
const int32 l_924 = l_arr_93[uint32(1)];
const int32 l_925 = l_924 + l_923;
l_arr_93[uint32(1)] = l_925;
const uint32 l_926 = l_723.i_1;
const uint32 l_927 = l_926 & uint32(252645135);
const uint32 l_928 = l_926 >> uint32(4);
const uint32 l_929 = l_928 & uint32(252645135);
const uint32 l_930 = l_887.i_1;
const int8_4 l_931 = reinterpret_cast<int8_4 const&>(l_927);
const int32_4 l_932 = int32_4{
int32(l_931.i_0),int32(l_931.i_1),int32(l_931.i_2),int32(l_931.i_3),};
const int8_4 l_933 = reinterpret_cast<int8_4 const&>(l_930);
const int32_4 l_934 = int32_4{
int32(l_933.i_0),int32(l_933.i_1),int32(l_933.i_2),int32(l_933.i_3),};
#if defined(__gfx1201__)
const int32 l_935 = __builtin_amdgcn_sudot4(true, l_927, true, l_930, 0, false);
#else
const int32 l_935 = l_932.i_0 * l_934.i_0 + l_932.i_1 * l_934.i_1 + l_932.i_2 * l_934.i_2 + l_932.i_3 * l_934.i_3;
#endif
const int32 l_936 = l_arr_90[uint32(1)];
const int32 l_937 = l_936 + l_935;
l_arr_90[uint32(1)] = l_937;
const uint32 l_938 = l_889.i_1;
const int8_4 l_939 = reinterpret_cast<int8_4 const&>(l_929);
const int32_4 l_940 = int32_4{
int32(l_939.i_0),int32(l_939.i_1),int32(l_939.i_2),int32(l_939.i_3),};
const int8_4 l_941 = reinterpret_cast<int8_4 const&>(l_938);
const int32_4 l_942 = int32_4{
int32(l_941.i_0),int32(l_941.i_1),int32(l_941.i_2),int32(l_941.i_3),};
#if defined(__gfx1201__)
const int32 l_943 = __builtin_amdgcn_sudot4(true, l_929, true, l_938, 0, false);
#else
const int32 l_943 = l_940.i_0 * l_942.i_0 + l_940.i_1 * l_942.i_1 + l_940.i_2 * l_942.i_2 + l_940.i_3 * l_942.i_3;
#endif
const int32 l_944 = l_arr_91[uint32(1)];
const int32 l_945 = l_944 + l_943;
l_arr_91[uint32(1)] = l_945;
const uint32 l_946 = l_887.i_1;
const uint32 l_947_bits = uint32(16843009);
const int8_4 l_947 = reinterpret_cast<int8_4 const&>(l_947_bits);
const int32_4 l_948 = int32_4{
int32(l_947.i_0),int32(l_947.i_1),int32(l_947.i_2),int32(l_947.i_3),};
const int8_4 l_949 = reinterpret_cast<int8_4 const&>(l_946);
const int32_4 l_950 = int32_4{
int32(l_949.i_0),int32(l_949.i_1),int32(l_949.i_2),int32(l_949.i_3),};
#if defined(__gfx1201__)
const int32 l_951 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_946, 0, false);
#else
const int32 l_951 = l_948.i_0 * l_950.i_0 + l_948.i_1 * l_950.i_1 + l_948.i_2 * l_950.i_2 + l_948.i_3 * l_950.i_3;
#endif
const int32 l_952 = l_arr_92[uint32(1)];
const int32 l_953 = l_952 + l_951;
l_arr_92[uint32(1)] = l_953;
const uint32 l_954 = l_889.i_1;
const uint32 l_955_bits = uint32(16843009);
const int8_4 l_955 = reinterpret_cast<int8_4 const&>(l_955_bits);
const int32_4 l_956 = int32_4{
int32(l_955.i_0),int32(l_955.i_1),int32(l_955.i_2),int32(l_955.i_3),};
const int8_4 l_957 = reinterpret_cast<int8_4 const&>(l_954);
const int32_4 l_958 = int32_4{
int32(l_957.i_0),int32(l_957.i_1),int32(l_957.i_2),int32(l_957.i_3),};
#if defined(__gfx1201__)
const int32 l_959 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_954, 0, false);
#else
const int32 l_959 = l_956.i_0 * l_958.i_0 + l_956.i_1 * l_958.i_1 + l_956.i_2 * l_958.i_2 + l_956.i_3 * l_958.i_3;
#endif
const int32 l_960 = l_arr_93[uint32(1)];
const int32 l_961 = l_960 + l_959;
l_arr_93[uint32(1)] = l_961;
const uint32 l_962 = l_723.i_2;
const uint32 l_963 = l_962 & uint32(252645135);
const uint32 l_964 = l_962 >> uint32(4);
const uint32 l_965 = l_964 & uint32(252645135);
const uint32 l_966 = l_887.i_2;
const int8_4 l_967 = reinterpret_cast<int8_4 const&>(l_963);
const int32_4 l_968 = int32_4{
int32(l_967.i_0),int32(l_967.i_1),int32(l_967.i_2),int32(l_967.i_3),};
const int8_4 l_969 = reinterpret_cast<int8_4 const&>(l_966);
const int32_4 l_970 = int32_4{
int32(l_969.i_0),int32(l_969.i_1),int32(l_969.i_2),int32(l_969.i_3),};
#if defined(__gfx1201__)
const int32 l_971 = __builtin_amdgcn_sudot4(true, l_963, true, l_966, 0, false);
#else
const int32 l_971 = l_968.i_0 * l_970.i_0 + l_968.i_1 * l_970.i_1 + l_968.i_2 * l_970.i_2 + l_968.i_3 * l_970.i_3;
#endif
const int32 l_972 = l_arr_90[uint32(1)];
const int32 l_973 = l_972 + l_971;
l_arr_90[uint32(1)] = l_973;
const uint32 l_974 = l_889.i_2;
const int8_4 l_975 = reinterpret_cast<int8_4 const&>(l_965);
const int32_4 l_976 = int32_4{
int32(l_975.i_0),int32(l_975.i_1),int32(l_975.i_2),int32(l_975.i_3),};
const int8_4 l_977 = reinterpret_cast<int8_4 const&>(l_974);
const int32_4 l_978 = int32_4{
int32(l_977.i_0),int32(l_977.i_1),int32(l_977.i_2),int32(l_977.i_3),};
#if defined(__gfx1201__)
const int32 l_979 = __builtin_amdgcn_sudot4(true, l_965, true, l_974, 0, false);
#else
const int32 l_979 = l_976.i_0 * l_978.i_0 + l_976.i_1 * l_978.i_1 + l_976.i_2 * l_978.i_2 + l_976.i_3 * l_978.i_3;
#endif
const int32 l_980 = l_arr_91[uint32(1)];
const int32 l_981 = l_980 + l_979;
l_arr_91[uint32(1)] = l_981;
const uint32 l_982 = l_887.i_2;
const uint32 l_983_bits = uint32(16843009);
const int8_4 l_983 = reinterpret_cast<int8_4 const&>(l_983_bits);
const int32_4 l_984 = int32_4{
int32(l_983.i_0),int32(l_983.i_1),int32(l_983.i_2),int32(l_983.i_3),};
const int8_4 l_985 = reinterpret_cast<int8_4 const&>(l_982);
const int32_4 l_986 = int32_4{
int32(l_985.i_0),int32(l_985.i_1),int32(l_985.i_2),int32(l_985.i_3),};
#if defined(__gfx1201__)
const int32 l_987 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_982, 0, false);
#else
const int32 l_987 = l_984.i_0 * l_986.i_0 + l_984.i_1 * l_986.i_1 + l_984.i_2 * l_986.i_2 + l_984.i_3 * l_986.i_3;
#endif
const int32 l_988 = l_arr_92[uint32(1)];
const int32 l_989 = l_988 + l_987;
l_arr_92[uint32(1)] = l_989;
const uint32 l_990 = l_889.i_2;
const uint32 l_991_bits = uint32(16843009);
const int8_4 l_991 = reinterpret_cast<int8_4 const&>(l_991_bits);
const int32_4 l_992 = int32_4{
int32(l_991.i_0),int32(l_991.i_1),int32(l_991.i_2),int32(l_991.i_3),};
const int8_4 l_993 = reinterpret_cast<int8_4 const&>(l_990);
const int32_4 l_994 = int32_4{
int32(l_993.i_0),int32(l_993.i_1),int32(l_993.i_2),int32(l_993.i_3),};
#if defined(__gfx1201__)
const int32 l_995 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_990, 0, false);
#else
const int32 l_995 = l_992.i_0 * l_994.i_0 + l_992.i_1 * l_994.i_1 + l_992.i_2 * l_994.i_2 + l_992.i_3 * l_994.i_3;
#endif
const int32 l_996 = l_arr_93[uint32(1)];
const int32 l_997 = l_996 + l_995;
l_arr_93[uint32(1)] = l_997;
const uint32 l_998 = l_723.i_3;
const uint32 l_999 = l_998 & uint32(252645135);
const uint32 l_1000 = l_998 >> uint32(4);
const uint32 l_1001 = l_1000 & uint32(252645135);
const uint32 l_1002 = l_887.i_3;
const int8_4 l_1003 = reinterpret_cast<int8_4 const&>(l_999);
const int32_4 l_1004 = int32_4{
int32(l_1003.i_0),int32(l_1003.i_1),int32(l_1003.i_2),int32(l_1003.i_3),};
const int8_4 l_1005 = reinterpret_cast<int8_4 const&>(l_1002);
const int32_4 l_1006 = int32_4{
int32(l_1005.i_0),int32(l_1005.i_1),int32(l_1005.i_2),int32(l_1005.i_3),};
#if defined(__gfx1201__)
const int32 l_1007 = __builtin_amdgcn_sudot4(true, l_999, true, l_1002, 0, false);
#else
const int32 l_1007 = l_1004.i_0 * l_1006.i_0 + l_1004.i_1 * l_1006.i_1 + l_1004.i_2 * l_1006.i_2 + l_1004.i_3 * l_1006.i_3;
#endif
const int32 l_1008 = l_arr_90[uint32(1)];
const int32 l_1009 = l_1008 + l_1007;
l_arr_90[uint32(1)] = l_1009;
const uint32 l_1010 = l_889.i_3;
const int8_4 l_1011 = reinterpret_cast<int8_4 const&>(l_1001);
const int32_4 l_1012 = int32_4{
int32(l_1011.i_0),int32(l_1011.i_1),int32(l_1011.i_2),int32(l_1011.i_3),};
const int8_4 l_1013 = reinterpret_cast<int8_4 const&>(l_1010);
const int32_4 l_1014 = int32_4{
int32(l_1013.i_0),int32(l_1013.i_1),int32(l_1013.i_2),int32(l_1013.i_3),};
#if defined(__gfx1201__)
const int32 l_1015 = __builtin_amdgcn_sudot4(true, l_1001, true, l_1010, 0, false);
#else
const int32 l_1015 = l_1012.i_0 * l_1014.i_0 + l_1012.i_1 * l_1014.i_1 + l_1012.i_2 * l_1014.i_2 + l_1012.i_3 * l_1014.i_3;
#endif
const int32 l_1016 = l_arr_91[uint32(1)];
const int32 l_1017 = l_1016 + l_1015;
l_arr_91[uint32(1)] = l_1017;
const uint32 l_1018 = l_887.i_3;
const uint32 l_1019_bits = uint32(16843009);
const int8_4 l_1019 = reinterpret_cast<int8_4 const&>(l_1019_bits);
const int32_4 l_1020 = int32_4{
int32(l_1019.i_0),int32(l_1019.i_1),int32(l_1019.i_2),int32(l_1019.i_3),};
const int8_4 l_1021 = reinterpret_cast<int8_4 const&>(l_1018);
const int32_4 l_1022 = int32_4{
int32(l_1021.i_0),int32(l_1021.i_1),int32(l_1021.i_2),int32(l_1021.i_3),};
#if defined(__gfx1201__)
const int32 l_1023 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1018, 0, false);
#else
const int32 l_1023 = l_1020.i_0 * l_1022.i_0 + l_1020.i_1 * l_1022.i_1 + l_1020.i_2 * l_1022.i_2 + l_1020.i_3 * l_1022.i_3;
#endif
const int32 l_1024 = l_arr_92[uint32(1)];
const int32 l_1025 = l_1024 + l_1023;
l_arr_92[uint32(1)] = l_1025;
const uint32 l_1026 = l_889.i_3;
const uint32 l_1027_bits = uint32(16843009);
const int8_4 l_1027 = reinterpret_cast<int8_4 const&>(l_1027_bits);
const int32_4 l_1028 = int32_4{
int32(l_1027.i_0),int32(l_1027.i_1),int32(l_1027.i_2),int32(l_1027.i_3),};
const int8_4 l_1029 = reinterpret_cast<int8_4 const&>(l_1026);
const int32_4 l_1030 = int32_4{
int32(l_1029.i_0),int32(l_1029.i_1),int32(l_1029.i_2),int32(l_1029.i_3),};
#if defined(__gfx1201__)
const int32 l_1031 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1026, 0, false);
#else
const int32 l_1031 = l_1028.i_0 * l_1030.i_0 + l_1028.i_1 * l_1030.i_1 + l_1028.i_2 * l_1030.i_2 + l_1028.i_3 * l_1030.i_3;
#endif
const int32 l_1032 = l_arr_93[uint32(1)];
const int32 l_1033 = l_1032 + l_1031;
l_arr_93[uint32(1)] = l_1033;
const uint32 l_1034 = l_arr_28[uint32(2)];
const uint32 l_1035 = l_1 / uint32(16);
const uint32 l_1036 = l_1034 * l_1035;
const uint32 l_1037 = l_46 * uint32(16);
const uint32 l_1038 = l_1036 + l_1037;
const uint32 l_1039 = l_47 * uint32(4);
const uint32 l_1040 = l_1038 + l_1039;
const uint32 l_1041 = l_1040 + uint32(1);
const uint32_4 l_1042 = buffer_1[l_1041];
const uint32 l_1043 = l_1041 + uint32(2);
const uint32_4 l_1044 = buffer_1[l_1043];
const uint32 l_1045 = l_723.i_0;
const uint32 l_1046 = l_1045 & uint32(252645135);
const uint32 l_1047 = l_1045 >> uint32(4);
const uint32 l_1048 = l_1047 & uint32(252645135);
const uint32 l_1049 = l_1042.i_0;
const int8_4 l_1050 = reinterpret_cast<int8_4 const&>(l_1046);
const int32_4 l_1051 = int32_4{
int32(l_1050.i_0),int32(l_1050.i_1),int32(l_1050.i_2),int32(l_1050.i_3),};
const int8_4 l_1052 = reinterpret_cast<int8_4 const&>(l_1049);
const int32_4 l_1053 = int32_4{
int32(l_1052.i_0),int32(l_1052.i_1),int32(l_1052.i_2),int32(l_1052.i_3),};
#if defined(__gfx1201__)
const int32 l_1054 = __builtin_amdgcn_sudot4(true, l_1046, true, l_1049, 0, false);
#else
const int32 l_1054 = l_1051.i_0 * l_1053.i_0 + l_1051.i_1 * l_1053.i_1 + l_1051.i_2 * l_1053.i_2 + l_1051.i_3 * l_1053.i_3;
#endif
const int32 l_1055 = l_arr_90[uint32(2)];
const int32 l_1056 = l_1055 + l_1054;
l_arr_90[uint32(2)] = l_1056;
const uint32 l_1057 = l_1044.i_0;
const int8_4 l_1058 = reinterpret_cast<int8_4 const&>(l_1048);
const int32_4 l_1059 = int32_4{
int32(l_1058.i_0),int32(l_1058.i_1),int32(l_1058.i_2),int32(l_1058.i_3),};
const int8_4 l_1060 = reinterpret_cast<int8_4 const&>(l_1057);
const int32_4 l_1061 = int32_4{
int32(l_1060.i_0),int32(l_1060.i_1),int32(l_1060.i_2),int32(l_1060.i_3),};
#if defined(__gfx1201__)
const int32 l_1062 = __builtin_amdgcn_sudot4(true, l_1048, true, l_1057, 0, false);
#else
const int32 l_1062 = l_1059.i_0 * l_1061.i_0 + l_1059.i_1 * l_1061.i_1 + l_1059.i_2 * l_1061.i_2 + l_1059.i_3 * l_1061.i_3;
#endif
const int32 l_1063 = l_arr_91[uint32(2)];
const int32 l_1064 = l_1063 + l_1062;
l_arr_91[uint32(2)] = l_1064;
const uint32 l_1065 = l_1042.i_0;
const uint32 l_1066_bits = uint32(16843009);
const int8_4 l_1066 = reinterpret_cast<int8_4 const&>(l_1066_bits);
const int32_4 l_1067 = int32_4{
int32(l_1066.i_0),int32(l_1066.i_1),int32(l_1066.i_2),int32(l_1066.i_3),};
const int8_4 l_1068 = reinterpret_cast<int8_4 const&>(l_1065);
const int32_4 l_1069 = int32_4{
int32(l_1068.i_0),int32(l_1068.i_1),int32(l_1068.i_2),int32(l_1068.i_3),};
#if defined(__gfx1201__)
const int32 l_1070 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1065, 0, false);
#else
const int32 l_1070 = l_1067.i_0 * l_1069.i_0 + l_1067.i_1 * l_1069.i_1 + l_1067.i_2 * l_1069.i_2 + l_1067.i_3 * l_1069.i_3;
#endif
const int32 l_1071 = l_arr_92[uint32(2)];
const int32 l_1072 = l_1071 + l_1070;
l_arr_92[uint32(2)] = l_1072;
const uint32 l_1073 = l_1044.i_0;
const uint32 l_1074_bits = uint32(16843009);
const int8_4 l_1074 = reinterpret_cast<int8_4 const&>(l_1074_bits);
const int32_4 l_1075 = int32_4{
int32(l_1074.i_0),int32(l_1074.i_1),int32(l_1074.i_2),int32(l_1074.i_3),};
const int8_4 l_1076 = reinterpret_cast<int8_4 const&>(l_1073);
const int32_4 l_1077 = int32_4{
int32(l_1076.i_0),int32(l_1076.i_1),int32(l_1076.i_2),int32(l_1076.i_3),};
#if defined(__gfx1201__)
const int32 l_1078 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1073, 0, false);
#else
const int32 l_1078 = l_1075.i_0 * l_1077.i_0 + l_1075.i_1 * l_1077.i_1 + l_1075.i_2 * l_1077.i_2 + l_1075.i_3 * l_1077.i_3;
#endif
const int32 l_1079 = l_arr_93[uint32(2)];
const int32 l_1080 = l_1079 + l_1078;
l_arr_93[uint32(2)] = l_1080;
const uint32 l_1081 = l_723.i_1;
const uint32 l_1082 = l_1081 & uint32(252645135);
const uint32 l_1083 = l_1081 >> uint32(4);
const uint32 l_1084 = l_1083 & uint32(252645135);
const uint32 l_1085 = l_1042.i_1;
const int8_4 l_1086 = reinterpret_cast<int8_4 const&>(l_1082);
const int32_4 l_1087 = int32_4{
int32(l_1086.i_0),int32(l_1086.i_1),int32(l_1086.i_2),int32(l_1086.i_3),};
const int8_4 l_1088 = reinterpret_cast<int8_4 const&>(l_1085);
const int32_4 l_1089 = int32_4{
int32(l_1088.i_0),int32(l_1088.i_1),int32(l_1088.i_2),int32(l_1088.i_3),};
#if defined(__gfx1201__)
const int32 l_1090 = __builtin_amdgcn_sudot4(true, l_1082, true, l_1085, 0, false);
#else
const int32 l_1090 = l_1087.i_0 * l_1089.i_0 + l_1087.i_1 * l_1089.i_1 + l_1087.i_2 * l_1089.i_2 + l_1087.i_3 * l_1089.i_3;
#endif
const int32 l_1091 = l_arr_90[uint32(2)];
const int32 l_1092 = l_1091 + l_1090;
l_arr_90[uint32(2)] = l_1092;
const uint32 l_1093 = l_1044.i_1;
const int8_4 l_1094 = reinterpret_cast<int8_4 const&>(l_1084);
const int32_4 l_1095 = int32_4{
int32(l_1094.i_0),int32(l_1094.i_1),int32(l_1094.i_2),int32(l_1094.i_3),};
const int8_4 l_1096 = reinterpret_cast<int8_4 const&>(l_1093);
const int32_4 l_1097 = int32_4{
int32(l_1096.i_0),int32(l_1096.i_1),int32(l_1096.i_2),int32(l_1096.i_3),};
#if defined(__gfx1201__)
const int32 l_1098 = __builtin_amdgcn_sudot4(true, l_1084, true, l_1093, 0, false);
#else
const int32 l_1098 = l_1095.i_0 * l_1097.i_0 + l_1095.i_1 * l_1097.i_1 + l_1095.i_2 * l_1097.i_2 + l_1095.i_3 * l_1097.i_3;
#endif
const int32 l_1099 = l_arr_91[uint32(2)];
const int32 l_1100 = l_1099 + l_1098;
l_arr_91[uint32(2)] = l_1100;
const uint32 l_1101 = l_1042.i_1;
const uint32 l_1102_bits = uint32(16843009);
const int8_4 l_1102 = reinterpret_cast<int8_4 const&>(l_1102_bits);
const int32_4 l_1103 = int32_4{
int32(l_1102.i_0),int32(l_1102.i_1),int32(l_1102.i_2),int32(l_1102.i_3),};
const int8_4 l_1104 = reinterpret_cast<int8_4 const&>(l_1101);
const int32_4 l_1105 = int32_4{
int32(l_1104.i_0),int32(l_1104.i_1),int32(l_1104.i_2),int32(l_1104.i_3),};
#if defined(__gfx1201__)
const int32 l_1106 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1101, 0, false);
#else
const int32 l_1106 = l_1103.i_0 * l_1105.i_0 + l_1103.i_1 * l_1105.i_1 + l_1103.i_2 * l_1105.i_2 + l_1103.i_3 * l_1105.i_3;
#endif
const int32 l_1107 = l_arr_92[uint32(2)];
const int32 l_1108 = l_1107 + l_1106;
l_arr_92[uint32(2)] = l_1108;
const uint32 l_1109 = l_1044.i_1;
const uint32 l_1110_bits = uint32(16843009);
const int8_4 l_1110 = reinterpret_cast<int8_4 const&>(l_1110_bits);
const int32_4 l_1111 = int32_4{
int32(l_1110.i_0),int32(l_1110.i_1),int32(l_1110.i_2),int32(l_1110.i_3),};
const int8_4 l_1112 = reinterpret_cast<int8_4 const&>(l_1109);
const int32_4 l_1113 = int32_4{
int32(l_1112.i_0),int32(l_1112.i_1),int32(l_1112.i_2),int32(l_1112.i_3),};
#if defined(__gfx1201__)
const int32 l_1114 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1109, 0, false);
#else
const int32 l_1114 = l_1111.i_0 * l_1113.i_0 + l_1111.i_1 * l_1113.i_1 + l_1111.i_2 * l_1113.i_2 + l_1111.i_3 * l_1113.i_3;
#endif
const int32 l_1115 = l_arr_93[uint32(2)];
const int32 l_1116 = l_1115 + l_1114;
l_arr_93[uint32(2)] = l_1116;
const uint32 l_1117 = l_723.i_2;
const uint32 l_1118 = l_1117 & uint32(252645135);
const uint32 l_1119 = l_1117 >> uint32(4);
const uint32 l_1120 = l_1119 & uint32(252645135);
const uint32 l_1121 = l_1042.i_2;
const int8_4 l_1122 = reinterpret_cast<int8_4 const&>(l_1118);
const int32_4 l_1123 = int32_4{
int32(l_1122.i_0),int32(l_1122.i_1),int32(l_1122.i_2),int32(l_1122.i_3),};
const int8_4 l_1124 = reinterpret_cast<int8_4 const&>(l_1121);
const int32_4 l_1125 = int32_4{
int32(l_1124.i_0),int32(l_1124.i_1),int32(l_1124.i_2),int32(l_1124.i_3),};
#if defined(__gfx1201__)
const int32 l_1126 = __builtin_amdgcn_sudot4(true, l_1118, true, l_1121, 0, false);
#else
const int32 l_1126 = l_1123.i_0 * l_1125.i_0 + l_1123.i_1 * l_1125.i_1 + l_1123.i_2 * l_1125.i_2 + l_1123.i_3 * l_1125.i_3;
#endif
const int32 l_1127 = l_arr_90[uint32(2)];
const int32 l_1128 = l_1127 + l_1126;
l_arr_90[uint32(2)] = l_1128;
const uint32 l_1129 = l_1044.i_2;
const int8_4 l_1130 = reinterpret_cast<int8_4 const&>(l_1120);
const int32_4 l_1131 = int32_4{
int32(l_1130.i_0),int32(l_1130.i_1),int32(l_1130.i_2),int32(l_1130.i_3),};
const int8_4 l_1132 = reinterpret_cast<int8_4 const&>(l_1129);
const int32_4 l_1133 = int32_4{
int32(l_1132.i_0),int32(l_1132.i_1),int32(l_1132.i_2),int32(l_1132.i_3),};
#if defined(__gfx1201__)
const int32 l_1134 = __builtin_amdgcn_sudot4(true, l_1120, true, l_1129, 0, false);
#else
const int32 l_1134 = l_1131.i_0 * l_1133.i_0 + l_1131.i_1 * l_1133.i_1 + l_1131.i_2 * l_1133.i_2 + l_1131.i_3 * l_1133.i_3;
#endif
const int32 l_1135 = l_arr_91[uint32(2)];
const int32 l_1136 = l_1135 + l_1134;
l_arr_91[uint32(2)] = l_1136;
const uint32 l_1137 = l_1042.i_2;
const uint32 l_1138_bits = uint32(16843009);
const int8_4 l_1138 = reinterpret_cast<int8_4 const&>(l_1138_bits);
const int32_4 l_1139 = int32_4{
int32(l_1138.i_0),int32(l_1138.i_1),int32(l_1138.i_2),int32(l_1138.i_3),};
const int8_4 l_1140 = reinterpret_cast<int8_4 const&>(l_1137);
const int32_4 l_1141 = int32_4{
int32(l_1140.i_0),int32(l_1140.i_1),int32(l_1140.i_2),int32(l_1140.i_3),};
#if defined(__gfx1201__)
const int32 l_1142 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1137, 0, false);
#else
const int32 l_1142 = l_1139.i_0 * l_1141.i_0 + l_1139.i_1 * l_1141.i_1 + l_1139.i_2 * l_1141.i_2 + l_1139.i_3 * l_1141.i_3;
#endif
const int32 l_1143 = l_arr_92[uint32(2)];
const int32 l_1144 = l_1143 + l_1142;
l_arr_92[uint32(2)] = l_1144;
const uint32 l_1145 = l_1044.i_2;
const uint32 l_1146_bits = uint32(16843009);
const int8_4 l_1146 = reinterpret_cast<int8_4 const&>(l_1146_bits);
const int32_4 l_1147 = int32_4{
int32(l_1146.i_0),int32(l_1146.i_1),int32(l_1146.i_2),int32(l_1146.i_3),};
const int8_4 l_1148 = reinterpret_cast<int8_4 const&>(l_1145);
const int32_4 l_1149 = int32_4{
int32(l_1148.i_0),int32(l_1148.i_1),int32(l_1148.i_2),int32(l_1148.i_3),};
#if defined(__gfx1201__)
const int32 l_1150 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1145, 0, false);
#else
const int32 l_1150 = l_1147.i_0 * l_1149.i_0 + l_1147.i_1 * l_1149.i_1 + l_1147.i_2 * l_1149.i_2 + l_1147.i_3 * l_1149.i_3;
#endif
const int32 l_1151 = l_arr_93[uint32(2)];
const int32 l_1152 = l_1151 + l_1150;
l_arr_93[uint32(2)] = l_1152;
const uint32 l_1153 = l_723.i_3;
const uint32 l_1154 = l_1153 & uint32(252645135);
const uint32 l_1155 = l_1153 >> uint32(4);
const uint32 l_1156 = l_1155 & uint32(252645135);
const uint32 l_1157 = l_1042.i_3;
const int8_4 l_1158 = reinterpret_cast<int8_4 const&>(l_1154);
const int32_4 l_1159 = int32_4{
int32(l_1158.i_0),int32(l_1158.i_1),int32(l_1158.i_2),int32(l_1158.i_3),};
const int8_4 l_1160 = reinterpret_cast<int8_4 const&>(l_1157);
const int32_4 l_1161 = int32_4{
int32(l_1160.i_0),int32(l_1160.i_1),int32(l_1160.i_2),int32(l_1160.i_3),};
#if defined(__gfx1201__)
const int32 l_1162 = __builtin_amdgcn_sudot4(true, l_1154, true, l_1157, 0, false);
#else
const int32 l_1162 = l_1159.i_0 * l_1161.i_0 + l_1159.i_1 * l_1161.i_1 + l_1159.i_2 * l_1161.i_2 + l_1159.i_3 * l_1161.i_3;
#endif
const int32 l_1163 = l_arr_90[uint32(2)];
const int32 l_1164 = l_1163 + l_1162;
l_arr_90[uint32(2)] = l_1164;
const uint32 l_1165 = l_1044.i_3;
const int8_4 l_1166 = reinterpret_cast<int8_4 const&>(l_1156);
const int32_4 l_1167 = int32_4{
int32(l_1166.i_0),int32(l_1166.i_1),int32(l_1166.i_2),int32(l_1166.i_3),};
const int8_4 l_1168 = reinterpret_cast<int8_4 const&>(l_1165);
const int32_4 l_1169 = int32_4{
int32(l_1168.i_0),int32(l_1168.i_1),int32(l_1168.i_2),int32(l_1168.i_3),};
#if defined(__gfx1201__)
const int32 l_1170 = __builtin_amdgcn_sudot4(true, l_1156, true, l_1165, 0, false);
#else
const int32 l_1170 = l_1167.i_0 * l_1169.i_0 + l_1167.i_1 * l_1169.i_1 + l_1167.i_2 * l_1169.i_2 + l_1167.i_3 * l_1169.i_3;
#endif
const int32 l_1171 = l_arr_91[uint32(2)];
const int32 l_1172 = l_1171 + l_1170;
l_arr_91[uint32(2)] = l_1172;
const uint32 l_1173 = l_1042.i_3;
const uint32 l_1174_bits = uint32(16843009);
const int8_4 l_1174 = reinterpret_cast<int8_4 const&>(l_1174_bits);
const int32_4 l_1175 = int32_4{
int32(l_1174.i_0),int32(l_1174.i_1),int32(l_1174.i_2),int32(l_1174.i_3),};
const int8_4 l_1176 = reinterpret_cast<int8_4 const&>(l_1173);
const int32_4 l_1177 = int32_4{
int32(l_1176.i_0),int32(l_1176.i_1),int32(l_1176.i_2),int32(l_1176.i_3),};
#if defined(__gfx1201__)
const int32 l_1178 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1173, 0, false);
#else
const int32 l_1178 = l_1175.i_0 * l_1177.i_0 + l_1175.i_1 * l_1177.i_1 + l_1175.i_2 * l_1177.i_2 + l_1175.i_3 * l_1177.i_3;
#endif
const int32 l_1179 = l_arr_92[uint32(2)];
const int32 l_1180 = l_1179 + l_1178;
l_arr_92[uint32(2)] = l_1180;
const uint32 l_1181 = l_1044.i_3;
const uint32 l_1182_bits = uint32(16843009);
const int8_4 l_1182 = reinterpret_cast<int8_4 const&>(l_1182_bits);
const int32_4 l_1183 = int32_4{
int32(l_1182.i_0),int32(l_1182.i_1),int32(l_1182.i_2),int32(l_1182.i_3),};
const int8_4 l_1184 = reinterpret_cast<int8_4 const&>(l_1181);
const int32_4 l_1185 = int32_4{
int32(l_1184.i_0),int32(l_1184.i_1),int32(l_1184.i_2),int32(l_1184.i_3),};
#if defined(__gfx1201__)
const int32 l_1186 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1181, 0, false);
#else
const int32 l_1186 = l_1183.i_0 * l_1185.i_0 + l_1183.i_1 * l_1185.i_1 + l_1183.i_2 * l_1185.i_2 + l_1183.i_3 * l_1185.i_3;
#endif
const int32 l_1187 = l_arr_93[uint32(2)];
const int32 l_1188 = l_1187 + l_1186;
l_arr_93[uint32(2)] = l_1188;
const uint32 l_1189 = l_arr_28[uint32(3)];
const uint32 l_1190 = l_1 / uint32(16);
const uint32 l_1191 = l_1189 * l_1190;
const uint32 l_1192 = l_46 * uint32(16);
const uint32 l_1193 = l_1191 + l_1192;
const uint32 l_1194 = l_47 * uint32(4);
const uint32 l_1195 = l_1193 + l_1194;
const uint32 l_1196 = l_1195 + uint32(1);
const uint32_4 l_1197 = buffer_1[l_1196];
const uint32 l_1198 = l_1196 + uint32(2);
const uint32_4 l_1199 = buffer_1[l_1198];
const uint32 l_1200 = l_723.i_0;
const uint32 l_1201 = l_1200 & uint32(252645135);
const uint32 l_1202 = l_1200 >> uint32(4);
const uint32 l_1203 = l_1202 & uint32(252645135);
const uint32 l_1204 = l_1197.i_0;
const int8_4 l_1205 = reinterpret_cast<int8_4 const&>(l_1201);
const int32_4 l_1206 = int32_4{
int32(l_1205.i_0),int32(l_1205.i_1),int32(l_1205.i_2),int32(l_1205.i_3),};
const int8_4 l_1207 = reinterpret_cast<int8_4 const&>(l_1204);
const int32_4 l_1208 = int32_4{
int32(l_1207.i_0),int32(l_1207.i_1),int32(l_1207.i_2),int32(l_1207.i_3),};
#if defined(__gfx1201__)
const int32 l_1209 = __builtin_amdgcn_sudot4(true, l_1201, true, l_1204, 0, false);
#else
const int32 l_1209 = l_1206.i_0 * l_1208.i_0 + l_1206.i_1 * l_1208.i_1 + l_1206.i_2 * l_1208.i_2 + l_1206.i_3 * l_1208.i_3;
#endif
const int32 l_1210 = l_arr_90[uint32(3)];
const int32 l_1211 = l_1210 + l_1209;
l_arr_90[uint32(3)] = l_1211;
const uint32 l_1212 = l_1199.i_0;
const int8_4 l_1213 = reinterpret_cast<int8_4 const&>(l_1203);
const int32_4 l_1214 = int32_4{
int32(l_1213.i_0),int32(l_1213.i_1),int32(l_1213.i_2),int32(l_1213.i_3),};
const int8_4 l_1215 = reinterpret_cast<int8_4 const&>(l_1212);
const int32_4 l_1216 = int32_4{
int32(l_1215.i_0),int32(l_1215.i_1),int32(l_1215.i_2),int32(l_1215.i_3),};
#if defined(__gfx1201__)
const int32 l_1217 = __builtin_amdgcn_sudot4(true, l_1203, true, l_1212, 0, false);
#else
const int32 l_1217 = l_1214.i_0 * l_1216.i_0 + l_1214.i_1 * l_1216.i_1 + l_1214.i_2 * l_1216.i_2 + l_1214.i_3 * l_1216.i_3;
#endif
const int32 l_1218 = l_arr_91[uint32(3)];
const int32 l_1219 = l_1218 + l_1217;
l_arr_91[uint32(3)] = l_1219;
const uint32 l_1220 = l_1197.i_0;
const uint32 l_1221_bits = uint32(16843009);
const int8_4 l_1221 = reinterpret_cast<int8_4 const&>(l_1221_bits);
const int32_4 l_1222 = int32_4{
int32(l_1221.i_0),int32(l_1221.i_1),int32(l_1221.i_2),int32(l_1221.i_3),};
const int8_4 l_1223 = reinterpret_cast<int8_4 const&>(l_1220);
const int32_4 l_1224 = int32_4{
int32(l_1223.i_0),int32(l_1223.i_1),int32(l_1223.i_2),int32(l_1223.i_3),};
#if defined(__gfx1201__)
const int32 l_1225 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1220, 0, false);
#else
const int32 l_1225 = l_1222.i_0 * l_1224.i_0 + l_1222.i_1 * l_1224.i_1 + l_1222.i_2 * l_1224.i_2 + l_1222.i_3 * l_1224.i_3;
#endif
const int32 l_1226 = l_arr_92[uint32(3)];
const int32 l_1227 = l_1226 + l_1225;
l_arr_92[uint32(3)] = l_1227;
const uint32 l_1228 = l_1199.i_0;
const uint32 l_1229_bits = uint32(16843009);
const int8_4 l_1229 = reinterpret_cast<int8_4 const&>(l_1229_bits);
const int32_4 l_1230 = int32_4{
int32(l_1229.i_0),int32(l_1229.i_1),int32(l_1229.i_2),int32(l_1229.i_3),};
const int8_4 l_1231 = reinterpret_cast<int8_4 const&>(l_1228);
const int32_4 l_1232 = int32_4{
int32(l_1231.i_0),int32(l_1231.i_1),int32(l_1231.i_2),int32(l_1231.i_3),};
#if defined(__gfx1201__)
const int32 l_1233 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1228, 0, false);
#else
const int32 l_1233 = l_1230.i_0 * l_1232.i_0 + l_1230.i_1 * l_1232.i_1 + l_1230.i_2 * l_1232.i_2 + l_1230.i_3 * l_1232.i_3;
#endif
const int32 l_1234 = l_arr_93[uint32(3)];
const int32 l_1235 = l_1234 + l_1233;
l_arr_93[uint32(3)] = l_1235;
const uint32 l_1236 = l_723.i_1;
const uint32 l_1237 = l_1236 & uint32(252645135);
const uint32 l_1238 = l_1236 >> uint32(4);
const uint32 l_1239 = l_1238 & uint32(252645135);
const uint32 l_1240 = l_1197.i_1;
const int8_4 l_1241 = reinterpret_cast<int8_4 const&>(l_1237);
const int32_4 l_1242 = int32_4{
int32(l_1241.i_0),int32(l_1241.i_1),int32(l_1241.i_2),int32(l_1241.i_3),};
const int8_4 l_1243 = reinterpret_cast<int8_4 const&>(l_1240);
const int32_4 l_1244 = int32_4{
int32(l_1243.i_0),int32(l_1243.i_1),int32(l_1243.i_2),int32(l_1243.i_3),};
#if defined(__gfx1201__)
const int32 l_1245 = __builtin_amdgcn_sudot4(true, l_1237, true, l_1240, 0, false);
#else
const int32 l_1245 = l_1242.i_0 * l_1244.i_0 + l_1242.i_1 * l_1244.i_1 + l_1242.i_2 * l_1244.i_2 + l_1242.i_3 * l_1244.i_3;
#endif
const int32 l_1246 = l_arr_90[uint32(3)];
const int32 l_1247 = l_1246 + l_1245;
l_arr_90[uint32(3)] = l_1247;
const uint32 l_1248 = l_1199.i_1;
const int8_4 l_1249 = reinterpret_cast<int8_4 const&>(l_1239);
const int32_4 l_1250 = int32_4{
int32(l_1249.i_0),int32(l_1249.i_1),int32(l_1249.i_2),int32(l_1249.i_3),};
const int8_4 l_1251 = reinterpret_cast<int8_4 const&>(l_1248);
const int32_4 l_1252 = int32_4{
int32(l_1251.i_0),int32(l_1251.i_1),int32(l_1251.i_2),int32(l_1251.i_3),};
#if defined(__gfx1201__)
const int32 l_1253 = __builtin_amdgcn_sudot4(true, l_1239, true, l_1248, 0, false);
#else
const int32 l_1253 = l_1250.i_0 * l_1252.i_0 + l_1250.i_1 * l_1252.i_1 + l_1250.i_2 * l_1252.i_2 + l_1250.i_3 * l_1252.i_3;
#endif
const int32 l_1254 = l_arr_91[uint32(3)];
const int32 l_1255 = l_1254 + l_1253;
l_arr_91[uint32(3)] = l_1255;
const uint32 l_1256 = l_1197.i_1;
const uint32 l_1257_bits = uint32(16843009);
const int8_4 l_1257 = reinterpret_cast<int8_4 const&>(l_1257_bits);
const int32_4 l_1258 = int32_4{
int32(l_1257.i_0),int32(l_1257.i_1),int32(l_1257.i_2),int32(l_1257.i_3),};
const int8_4 l_1259 = reinterpret_cast<int8_4 const&>(l_1256);
const int32_4 l_1260 = int32_4{
int32(l_1259.i_0),int32(l_1259.i_1),int32(l_1259.i_2),int32(l_1259.i_3),};
#if defined(__gfx1201__)
const int32 l_1261 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1256, 0, false);
#else
const int32 l_1261 = l_1258.i_0 * l_1260.i_0 + l_1258.i_1 * l_1260.i_1 + l_1258.i_2 * l_1260.i_2 + l_1258.i_3 * l_1260.i_3;
#endif
const int32 l_1262 = l_arr_92[uint32(3)];
const int32 l_1263 = l_1262 + l_1261;
l_arr_92[uint32(3)] = l_1263;
const uint32 l_1264 = l_1199.i_1;
const uint32 l_1265_bits = uint32(16843009);
const int8_4 l_1265 = reinterpret_cast<int8_4 const&>(l_1265_bits);
const int32_4 l_1266 = int32_4{
int32(l_1265.i_0),int32(l_1265.i_1),int32(l_1265.i_2),int32(l_1265.i_3),};
const int8_4 l_1267 = reinterpret_cast<int8_4 const&>(l_1264);
const int32_4 l_1268 = int32_4{
int32(l_1267.i_0),int32(l_1267.i_1),int32(l_1267.i_2),int32(l_1267.i_3),};
#if defined(__gfx1201__)
const int32 l_1269 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1264, 0, false);
#else
const int32 l_1269 = l_1266.i_0 * l_1268.i_0 + l_1266.i_1 * l_1268.i_1 + l_1266.i_2 * l_1268.i_2 + l_1266.i_3 * l_1268.i_3;
#endif
const int32 l_1270 = l_arr_93[uint32(3)];
const int32 l_1271 = l_1270 + l_1269;
l_arr_93[uint32(3)] = l_1271;
const uint32 l_1272 = l_723.i_2;
const uint32 l_1273 = l_1272 & uint32(252645135);
const uint32 l_1274 = l_1272 >> uint32(4);
const uint32 l_1275 = l_1274 & uint32(252645135);
const uint32 l_1276 = l_1197.i_2;
const int8_4 l_1277 = reinterpret_cast<int8_4 const&>(l_1273);
const int32_4 l_1278 = int32_4{
int32(l_1277.i_0),int32(l_1277.i_1),int32(l_1277.i_2),int32(l_1277.i_3),};
const int8_4 l_1279 = reinterpret_cast<int8_4 const&>(l_1276);
const int32_4 l_1280 = int32_4{
int32(l_1279.i_0),int32(l_1279.i_1),int32(l_1279.i_2),int32(l_1279.i_3),};
#if defined(__gfx1201__)
const int32 l_1281 = __builtin_amdgcn_sudot4(true, l_1273, true, l_1276, 0, false);
#else
const int32 l_1281 = l_1278.i_0 * l_1280.i_0 + l_1278.i_1 * l_1280.i_1 + l_1278.i_2 * l_1280.i_2 + l_1278.i_3 * l_1280.i_3;
#endif
const int32 l_1282 = l_arr_90[uint32(3)];
const int32 l_1283 = l_1282 + l_1281;
l_arr_90[uint32(3)] = l_1283;
const uint32 l_1284 = l_1199.i_2;
const int8_4 l_1285 = reinterpret_cast<int8_4 const&>(l_1275);
const int32_4 l_1286 = int32_4{
int32(l_1285.i_0),int32(l_1285.i_1),int32(l_1285.i_2),int32(l_1285.i_3),};
const int8_4 l_1287 = reinterpret_cast<int8_4 const&>(l_1284);
const int32_4 l_1288 = int32_4{
int32(l_1287.i_0),int32(l_1287.i_1),int32(l_1287.i_2),int32(l_1287.i_3),};
#if defined(__gfx1201__)
const int32 l_1289 = __builtin_amdgcn_sudot4(true, l_1275, true, l_1284, 0, false);
#else
const int32 l_1289 = l_1286.i_0 * l_1288.i_0 + l_1286.i_1 * l_1288.i_1 + l_1286.i_2 * l_1288.i_2 + l_1286.i_3 * l_1288.i_3;
#endif
const int32 l_1290 = l_arr_91[uint32(3)];
const int32 l_1291 = l_1290 + l_1289;
l_arr_91[uint32(3)] = l_1291;
const uint32 l_1292 = l_1197.i_2;
const uint32 l_1293_bits = uint32(16843009);
const int8_4 l_1293 = reinterpret_cast<int8_4 const&>(l_1293_bits);
const int32_4 l_1294 = int32_4{
int32(l_1293.i_0),int32(l_1293.i_1),int32(l_1293.i_2),int32(l_1293.i_3),};
const int8_4 l_1295 = reinterpret_cast<int8_4 const&>(l_1292);
const int32_4 l_1296 = int32_4{
int32(l_1295.i_0),int32(l_1295.i_1),int32(l_1295.i_2),int32(l_1295.i_3),};
#if defined(__gfx1201__)
const int32 l_1297 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1292, 0, false);
#else
const int32 l_1297 = l_1294.i_0 * l_1296.i_0 + l_1294.i_1 * l_1296.i_1 + l_1294.i_2 * l_1296.i_2 + l_1294.i_3 * l_1296.i_3;
#endif
const int32 l_1298 = l_arr_92[uint32(3)];
const int32 l_1299 = l_1298 + l_1297;
l_arr_92[uint32(3)] = l_1299;
const uint32 l_1300 = l_1199.i_2;
const uint32 l_1301_bits = uint32(16843009);
const int8_4 l_1301 = reinterpret_cast<int8_4 const&>(l_1301_bits);
const int32_4 l_1302 = int32_4{
int32(l_1301.i_0),int32(l_1301.i_1),int32(l_1301.i_2),int32(l_1301.i_3),};
const int8_4 l_1303 = reinterpret_cast<int8_4 const&>(l_1300);
const int32_4 l_1304 = int32_4{
int32(l_1303.i_0),int32(l_1303.i_1),int32(l_1303.i_2),int32(l_1303.i_3),};
#if defined(__gfx1201__)
const int32 l_1305 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1300, 0, false);
#else
const int32 l_1305 = l_1302.i_0 * l_1304.i_0 + l_1302.i_1 * l_1304.i_1 + l_1302.i_2 * l_1304.i_2 + l_1302.i_3 * l_1304.i_3;
#endif
const int32 l_1306 = l_arr_93[uint32(3)];
const int32 l_1307 = l_1306 + l_1305;
l_arr_93[uint32(3)] = l_1307;
const uint32 l_1308 = l_723.i_3;
const uint32 l_1309 = l_1308 & uint32(252645135);
const uint32 l_1310 = l_1308 >> uint32(4);
const uint32 l_1311 = l_1310 & uint32(252645135);
const uint32 l_1312 = l_1197.i_3;
const int8_4 l_1313 = reinterpret_cast<int8_4 const&>(l_1309);
const int32_4 l_1314 = int32_4{
int32(l_1313.i_0),int32(l_1313.i_1),int32(l_1313.i_2),int32(l_1313.i_3),};
const int8_4 l_1315 = reinterpret_cast<int8_4 const&>(l_1312);
const int32_4 l_1316 = int32_4{
int32(l_1315.i_0),int32(l_1315.i_1),int32(l_1315.i_2),int32(l_1315.i_3),};
#if defined(__gfx1201__)
const int32 l_1317 = __builtin_amdgcn_sudot4(true, l_1309, true, l_1312, 0, false);
#else
const int32 l_1317 = l_1314.i_0 * l_1316.i_0 + l_1314.i_1 * l_1316.i_1 + l_1314.i_2 * l_1316.i_2 + l_1314.i_3 * l_1316.i_3;
#endif
const int32 l_1318 = l_arr_90[uint32(3)];
const int32 l_1319 = l_1318 + l_1317;
l_arr_90[uint32(3)] = l_1319;
const uint32 l_1320 = l_1199.i_3;
const int8_4 l_1321 = reinterpret_cast<int8_4 const&>(l_1311);
const int32_4 l_1322 = int32_4{
int32(l_1321.i_0),int32(l_1321.i_1),int32(l_1321.i_2),int32(l_1321.i_3),};
const int8_4 l_1323 = reinterpret_cast<int8_4 const&>(l_1320);
const int32_4 l_1324 = int32_4{
int32(l_1323.i_0),int32(l_1323.i_1),int32(l_1323.i_2),int32(l_1323.i_3),};
#if defined(__gfx1201__)
const int32 l_1325 = __builtin_amdgcn_sudot4(true, l_1311, true, l_1320, 0, false);
#else
const int32 l_1325 = l_1322.i_0 * l_1324.i_0 + l_1322.i_1 * l_1324.i_1 + l_1322.i_2 * l_1324.i_2 + l_1322.i_3 * l_1324.i_3;
#endif
const int32 l_1326 = l_arr_91[uint32(3)];
const int32 l_1327 = l_1326 + l_1325;
l_arr_91[uint32(3)] = l_1327;
const uint32 l_1328 = l_1197.i_3;
const uint32 l_1329_bits = uint32(16843009);
const int8_4 l_1329 = reinterpret_cast<int8_4 const&>(l_1329_bits);
const int32_4 l_1330 = int32_4{
int32(l_1329.i_0),int32(l_1329.i_1),int32(l_1329.i_2),int32(l_1329.i_3),};
const int8_4 l_1331 = reinterpret_cast<int8_4 const&>(l_1328);
const int32_4 l_1332 = int32_4{
int32(l_1331.i_0),int32(l_1331.i_1),int32(l_1331.i_2),int32(l_1331.i_3),};
#if defined(__gfx1201__)
const int32 l_1333 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1328, 0, false);
#else
const int32 l_1333 = l_1330.i_0 * l_1332.i_0 + l_1330.i_1 * l_1332.i_1 + l_1330.i_2 * l_1332.i_2 + l_1330.i_3 * l_1332.i_3;
#endif
const int32 l_1334 = l_arr_92[uint32(3)];
const int32 l_1335 = l_1334 + l_1333;
l_arr_92[uint32(3)] = l_1335;
const uint32 l_1336 = l_1199.i_3;
const uint32 l_1337_bits = uint32(16843009);
const int8_4 l_1337 = reinterpret_cast<int8_4 const&>(l_1337_bits);
const int32_4 l_1338 = int32_4{
int32(l_1337.i_0),int32(l_1337.i_1),int32(l_1337.i_2),int32(l_1337.i_3),};
const int8_4 l_1339 = reinterpret_cast<int8_4 const&>(l_1336);
const int32_4 l_1340 = int32_4{
int32(l_1339.i_0),int32(l_1339.i_1),int32(l_1339.i_2),int32(l_1339.i_3),};
#if defined(__gfx1201__)
const int32 l_1341 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1336, 0, false);
#else
const int32 l_1341 = l_1338.i_0 * l_1340.i_0 + l_1338.i_1 * l_1340.i_1 + l_1338.i_2 * l_1340.i_2 + l_1338.i_3 * l_1340.i_3;
#endif
const int32 l_1342 = l_arr_93[uint32(3)];
const int32 l_1343 = l_1342 + l_1341;
l_arr_93[uint32(3)] = l_1343;
const uint32 l_1344 = l_arr_28[uint32(0)];
const uint32 l_1345 = l_1 / uint32(32);
const uint32 l_1346 = l_1344 * l_1345;
const uint32 l_1347 = l_46 * uint32(8);
const uint32 l_1348 = l_1346 + l_1347;
const uint32 l_1349 = l_47 * uint32(2);
const uint32 l_1350 = l_1348 + l_1349;
const __half l_1351 = buffer_2[l_1350];
const float l_1352 = float(l_1351);
const uint32 l_1353 = l_1350 + uint32(1);
const __half l_1354 = buffer_2[l_1353];
const float l_1355 = float(l_1354);
const int32 l_1356 = l_arr_90[uint32(0)];
const int32 l_1357 = l_1356 * l_81;
const float l_1358 = float(l_1357);
const int32 l_1359 = l_arr_91[uint32(0)];
const int32 l_1360 = l_1359 * l_84;
const float l_1361 = float(l_1360);
const float l_1362 = l_1355 * l_1361;
const float l_1363 = fma(l_1352, l_1358, l_1362);
const int32 l_1364 = l_arr_92[uint32(0)];
const int32 l_1365 = l_1364 * l_86;
const float l_1366 = float(l_1365);
const int32 l_1367 = l_arr_93[uint32(0)];
const int32 l_1368 = l_1367 * l_89;
const float l_1369 = float(l_1368);
const float l_1370 = l_1355 * l_1369;
const float l_1371 = fma(l_1352, l_1366, l_1370);
const float l_1372 = l_57 * l_1371;
const float l_1373 = -l_1372;
const float l_1374 = fma(l_55, l_1363, l_1373);
const float l_1375 = l_arr_29[uint32(0)];
const float l_1376 = l_1375 + l_1374;
l_arr_29[uint32(0)] = l_1376;
const uint32 l_1377 = l_arr_28[uint32(1)];
const uint32 l_1378 = l_1 / uint32(32);
const uint32 l_1379 = l_1377 * l_1378;
const uint32 l_1380 = l_46 * uint32(8);
const uint32 l_1381 = l_1379 + l_1380;
const uint32 l_1382 = l_47 * uint32(2);
const uint32 l_1383 = l_1381 + l_1382;
const __half l_1384 = buffer_2[l_1383];
const float l_1385 = float(l_1384);
const uint32 l_1386 = l_1383 + uint32(1);
const __half l_1387 = buffer_2[l_1386];
const float l_1388 = float(l_1387);
const int32 l_1389 = l_arr_90[uint32(1)];
const int32 l_1390 = l_1389 * l_81;
const float l_1391 = float(l_1390);
const int32 l_1392 = l_arr_91[uint32(1)];
const int32 l_1393 = l_1392 * l_84;
const float l_1394 = float(l_1393);
const float l_1395 = l_1388 * l_1394;
const float l_1396 = fma(l_1385, l_1391, l_1395);
const int32 l_1397 = l_arr_92[uint32(1)];
const int32 l_1398 = l_1397 * l_86;
const float l_1399 = float(l_1398);
const int32 l_1400 = l_arr_93[uint32(1)];
const int32 l_1401 = l_1400 * l_89;
const float l_1402 = float(l_1401);
const float l_1403 = l_1388 * l_1402;
const float l_1404 = fma(l_1385, l_1399, l_1403);
const float l_1405 = l_57 * l_1404;
const float l_1406 = -l_1405;
const float l_1407 = fma(l_55, l_1396, l_1406);
const float l_1408 = l_arr_29[uint32(1)];
const float l_1409 = l_1408 + l_1407;
l_arr_29[uint32(1)] = l_1409;
const uint32 l_1410 = l_arr_28[uint32(2)];
const uint32 l_1411 = l_1 / uint32(32);
const uint32 l_1412 = l_1410 * l_1411;
const uint32 l_1413 = l_46 * uint32(8);
const uint32 l_1414 = l_1412 + l_1413;
const uint32 l_1415 = l_47 * uint32(2);
const uint32 l_1416 = l_1414 + l_1415;
const __half l_1417 = buffer_2[l_1416];
const float l_1418 = float(l_1417);
const uint32 l_1419 = l_1416 + uint32(1);
const __half l_1420 = buffer_2[l_1419];
const float l_1421 = float(l_1420);
const int32 l_1422 = l_arr_90[uint32(2)];
const int32 l_1423 = l_1422 * l_81;
const float l_1424 = float(l_1423);
const int32 l_1425 = l_arr_91[uint32(2)];
const int32 l_1426 = l_1425 * l_84;
const float l_1427 = float(l_1426);
const float l_1428 = l_1421 * l_1427;
const float l_1429 = fma(l_1418, l_1424, l_1428);
const int32 l_1430 = l_arr_92[uint32(2)];
const int32 l_1431 = l_1430 * l_86;
const float l_1432 = float(l_1431);
const int32 l_1433 = l_arr_93[uint32(2)];
const int32 l_1434 = l_1433 * l_89;
const float l_1435 = float(l_1434);
const float l_1436 = l_1421 * l_1435;
const float l_1437 = fma(l_1418, l_1432, l_1436);
const float l_1438 = l_57 * l_1437;
const float l_1439 = -l_1438;
const float l_1440 = fma(l_55, l_1429, l_1439);
const float l_1441 = l_arr_29[uint32(2)];
const float l_1442 = l_1441 + l_1440;
l_arr_29[uint32(2)] = l_1442;
const uint32 l_1443 = l_arr_28[uint32(3)];
const uint32 l_1444 = l_1 / uint32(32);
const uint32 l_1445 = l_1443 * l_1444;
const uint32 l_1446 = l_46 * uint32(8);
const uint32 l_1447 = l_1445 + l_1446;
const uint32 l_1448 = l_47 * uint32(2);
const uint32 l_1449 = l_1447 + l_1448;
const __half l_1450 = buffer_2[l_1449];
const float l_1451 = float(l_1450);
const uint32 l_1452 = l_1449 + uint32(1);
const __half l_1453 = buffer_2[l_1452];
const float l_1454 = float(l_1453);
const int32 l_1455 = l_arr_90[uint32(3)];
const int32 l_1456 = l_1455 * l_81;
const float l_1457 = float(l_1456);
const int32 l_1458 = l_arr_91[uint32(3)];
const int32 l_1459 = l_1458 * l_84;
const float l_1460 = float(l_1459);
const float l_1461 = l_1454 * l_1460;
const float l_1462 = fma(l_1451, l_1457, l_1461);
const int32 l_1463 = l_arr_92[uint32(3)];
const int32 l_1464 = l_1463 * l_86;
const float l_1465 = float(l_1464);
const int32 l_1466 = l_arr_93[uint32(3)];
const int32 l_1467 = l_1466 * l_89;
const float l_1468 = float(l_1467);
const float l_1469 = l_1454 * l_1468;
const float l_1470 = fma(l_1451, l_1465, l_1469);
const float l_1471 = l_57 * l_1470;
const float l_1472 = -l_1471;
const float l_1473 = fma(l_55, l_1462, l_1472);
const float l_1474 = l_arr_29[uint32(3)];
const float l_1475 = l_1474 + l_1473;
l_arr_29[uint32(3)] = l_1475;
l_mut_42 = l_mut_42 + uint32(32);
}
const float l_1476 = l_arr_29[uint32(0)];
const float l_1477 = { __shfl_down(l_1476, uint32(16)) };
const float l_1478 = l_arr_29[uint32(0)];
const float l_1479 = l_1478 + l_1477;
l_arr_29[uint32(0)] = l_1479;
const float l_1480 = l_arr_29[uint32(1)];
const float l_1481 = { __shfl_down(l_1480, uint32(16)) };
const float l_1482 = l_arr_29[uint32(1)];
const float l_1483 = l_1482 + l_1481;
l_arr_29[uint32(1)] = l_1483;
const float l_1484 = l_arr_29[uint32(2)];
const float l_1485 = { __shfl_down(l_1484, uint32(16)) };
const float l_1486 = l_arr_29[uint32(2)];
const float l_1487 = l_1486 + l_1485;
l_arr_29[uint32(2)] = l_1487;
const float l_1488 = l_arr_29[uint32(3)];
const float l_1489 = { __shfl_down(l_1488, uint32(16)) };
const float l_1490 = l_arr_29[uint32(3)];
const float l_1491 = l_1490 + l_1489;
l_arr_29[uint32(3)] = l_1491;
const float l_1492 = l_arr_29[uint32(0)];
const float l_1493 = { __shfl_down(l_1492, uint32(8)) };
const float l_1494 = l_arr_29[uint32(0)];
const float l_1495 = l_1494 + l_1493;
l_arr_29[uint32(0)] = l_1495;
const float l_1496 = l_arr_29[uint32(1)];
const float l_1497 = { __shfl_down(l_1496, uint32(8)) };
const float l_1498 = l_arr_29[uint32(1)];
const float l_1499 = l_1498 + l_1497;
l_arr_29[uint32(1)] = l_1499;
const float l_1500 = l_arr_29[uint32(2)];
const float l_1501 = { __shfl_down(l_1500, uint32(8)) };
const float l_1502 = l_arr_29[uint32(2)];
const float l_1503 = l_1502 + l_1501;
l_arr_29[uint32(2)] = l_1503;
const float l_1504 = l_arr_29[uint32(3)];
const float l_1505 = { __shfl_down(l_1504, uint32(8)) };
const float l_1506 = l_arr_29[uint32(3)];
const float l_1507 = l_1506 + l_1505;
l_arr_29[uint32(3)] = l_1507;
const float l_1508 = l_arr_29[uint32(0)];
const float l_1509 = { __shfl_down(l_1508, uint32(4)) };
const float l_1510 = l_arr_29[uint32(0)];
const float l_1511 = l_1510 + l_1509;
l_arr_29[uint32(0)] = l_1511;
const float l_1512 = l_arr_29[uint32(1)];
const float l_1513 = { __shfl_down(l_1512, uint32(4)) };
const float l_1514 = l_arr_29[uint32(1)];
const float l_1515 = l_1514 + l_1513;
l_arr_29[uint32(1)] = l_1515;
const float l_1516 = l_arr_29[uint32(2)];
const float l_1517 = { __shfl_down(l_1516, uint32(4)) };
const float l_1518 = l_arr_29[uint32(2)];
const float l_1519 = l_1518 + l_1517;
l_arr_29[uint32(2)] = l_1519;
const float l_1520 = l_arr_29[uint32(3)];
const float l_1521 = { __shfl_down(l_1520, uint32(4)) };
const float l_1522 = l_arr_29[uint32(3)];
const float l_1523 = l_1522 + l_1521;
l_arr_29[uint32(3)] = l_1523;
const float l_1524 = l_arr_29[uint32(0)];
const float l_1525 = { __shfl_down(l_1524, uint32(2)) };
const float l_1526 = l_arr_29[uint32(0)];
const float l_1527 = l_1526 + l_1525;
l_arr_29[uint32(0)] = l_1527;
const float l_1528 = l_arr_29[uint32(1)];
const float l_1529 = { __shfl_down(l_1528, uint32(2)) };
const float l_1530 = l_arr_29[uint32(1)];
const float l_1531 = l_1530 + l_1529;
l_arr_29[uint32(1)] = l_1531;
const float l_1532 = l_arr_29[uint32(2)];
const float l_1533 = { __shfl_down(l_1532, uint32(2)) };
const float l_1534 = l_arr_29[uint32(2)];
const float l_1535 = l_1534 + l_1533;
l_arr_29[uint32(2)] = l_1535;
const float l_1536 = l_arr_29[uint32(3)];
const float l_1537 = { __shfl_down(l_1536, uint32(2)) };
const float l_1538 = l_arr_29[uint32(3)];
const float l_1539 = l_1538 + l_1537;
l_arr_29[uint32(3)] = l_1539;
const float l_1540 = l_arr_29[uint32(0)];
const float l_1541 = { __shfl_down(l_1540, uint32(1)) };
const float l_1542 = l_arr_29[uint32(0)];
const float l_1543 = l_1542 + l_1541;
l_arr_29[uint32(0)] = l_1543;
const float l_1544 = l_arr_29[uint32(1)];
const float l_1545 = { __shfl_down(l_1544, uint32(1)) };
const float l_1546 = l_arr_29[uint32(1)];
const float l_1547 = l_1546 + l_1545;
l_arr_29[uint32(1)] = l_1547;
const float l_1548 = l_arr_29[uint32(2)];
const float l_1549 = { __shfl_down(l_1548, uint32(1)) };
const float l_1550 = l_arr_29[uint32(2)];
const float l_1551 = l_1550 + l_1549;
l_arr_29[uint32(2)] = l_1551;
const float l_1552 = l_arr_29[uint32(3)];
const float l_1553 = { __shfl_down(l_1552, uint32(1)) };
const float l_1554 = l_arr_29[uint32(3)];
const float l_1555 = l_1554 + l_1553;
l_arr_29[uint32(3)] = l_1555;
const bool l_1556 = l_13 == uint32(0);
if (l_1556) {
const uint32 l_1557 = l_arr_28[uint32(0)];
const uint32 l_1558 = l_1557 * l_0;
const uint32 l_1559 = l_1558 + l_7;
const float l_1560 = l_arr_29[uint32(0)];
const __half l_1561 = __half(l_1560);
buffer_9[l_1559] = l_1561;
const uint32 l_1562 = l_arr_28[uint32(1)];
const uint32 l_1563 = l_1562 * l_0;
const uint32 l_1564 = l_1563 + l_7;
const float l_1565 = l_arr_29[uint32(1)];
const __half l_1566 = __half(l_1565);
buffer_9[l_1564] = l_1566;
const uint32 l_1567 = l_arr_28[uint32(2)];
const uint32 l_1568 = l_1567 * l_0;
const uint32 l_1569 = l_1568 + l_7;
const float l_1570 = l_arr_29[uint32(2)];
const __half l_1571 = __half(l_1570);
buffer_9[l_1569] = l_1571;
const uint32 l_1572 = l_arr_28[uint32(3)];
const uint32 l_1573 = l_1572 * l_0;
const uint32 l_1574 = l_1573 + l_7;
const float l_1575 = l_arr_29[uint32(3)];
const __half l_1576 = __half(l_1575);
buffer_9[l_1574] = l_1576;
}
l_mut_22 = l_mut_22 + uint32(4);
}
while (true) {
const bool l_1577 = l_mut_22 < l_21;
const bool l_1578 = !l_1577;
if (l_1578) {
break;}
const uint32 l_1579 = l_19 + l_mut_22;
const uint32 l_1580 = l_1 / uint32(256);
const uint32 l_1583 = l_1579 + uint32(0);
const int32 l_1584 = buffer_3[l_1583];
const uint32 l_1585 = uint32(l_1584);
l_arr_1581[uint32(0)] = l_1585;
l_arr_1582[uint32(0)] = float(0.0);
l_mut_67 = l_13;
while (true) {
const uint32 l_1586 = l_1580 * uint32(4);
const bool l_1587 = l_mut_67 < l_1586;
const bool l_1588 = !l_1587;
if (l_1588) {
break;}
const uint32 l_1589 = l_mut_67 / uint32(4);
const uint32 l_1590 = l_mut_67 % uint32(4);
const uint32 l_1591 = l_17 * l_1580;
const uint32 l_1592 = l_1591 + l_1589;
const uint32 l_1593 = l_1592 * uint32(9);
const uint32_4 l_1594 = buffer_0[l_1593];
const uint32 l_1595 = l_1594.i_0;
const __half_2 l_1596 = reinterpret_cast<__half_2 const&>(l_1595);
const __half l_1597 = l_1596.i_0;
const float l_1598 = float(l_1597);
const __half l_1599 = l_1596.i_1;
const float l_1600 = float(l_1599);
const uint32 l_1601 = l_1590 % uint32(2);
const uint32 l_1602 = l_1601 * uint32(16);
const uint32 l_1603 = l_1594.i_1;
const uint32 l_1604 = l_1603 >> l_1602;
const uint32 l_1605 = l_1594.i_2;
const uint32 l_1606 = l_1605 >> l_1602;
const uint32 l_1607 = l_1604 & uint32(16191);
l_mut_65 = l_1607;
const uint32 l_1608 = l_1606 & uint32(16191);
l_mut_42 = l_1608;
const bool l_1609 = l_1590 >= uint32(2);
if (l_1609) {
const uint32 l_1610 = l_1594.i_3;
const uint32 l_1611 = l_1610 >> l_1602;
const uint32 l_1612 = l_1611 & uint32(3855);
const uint32 l_1613 = l_1604 >> uint32(2);
const uint32 l_1614 = l_1613 & uint32(12336);
const uint32 l_1615 = l_1612 | l_1614;
l_mut_65 = l_1615;
const uint32 l_1616 = l_1611 >> uint32(4);
const uint32 l_1617 = l_1616 & uint32(3855);
const uint32 l_1618 = l_1606 >> uint32(2);
const uint32 l_1619 = l_1618 & uint32(12336);
const uint32 l_1620 = l_1617 | l_1619;
l_mut_42 = l_1620;
}
const uint32 l_1621 = l_mut_65 & uint32(63);
const int32 l_1622 = int32(l_1621);
const uint32 l_1623 = l_mut_65 >> uint32(8);
const uint32 l_1624 = l_1623 & uint32(63);
const int32 l_1625 = int32(l_1624);
const uint32 l_1626 = l_mut_42 & uint32(63);
const int32 l_1627 = int32(l_1626);
const uint32 l_1628 = l_mut_42 >> uint32(8);
const uint32 l_1629 = l_1628 & uint32(63);
const int32 l_1630 = int32(l_1629);
l_arr_1631[uint32(0)] = int32(0);
l_arr_1632[uint32(0)] = int32(0);
l_arr_1633[uint32(0)] = int32(0);
l_arr_1634[uint32(0)] = int32(0);
const uint32 l_1635 = l_1593 + uint32(1);
const uint32 l_1636 = l_1590 * uint32(2);
const uint32 l_1637 = l_1635 + l_1636;
const uint32 l_1638 = l_1637 + uint32(0);
const uint32_4 l_1639 = buffer_0[l_1638];
const uint32 l_1640 = l_arr_1581[uint32(0)];
const uint32 l_1641 = l_1 / uint32(16);
const uint32 l_1642 = l_1640 * l_1641;
const uint32 l_1643 = l_1589 * uint32(16);
const uint32 l_1644 = l_1642 + l_1643;
const uint32 l_1645 = l_1590 * uint32(4);
const uint32 l_1646 = l_1644 + l_1645;
const uint32 l_1647 = l_1646 + uint32(0);
const uint32_4 l_1648 = buffer_1[l_1647];
const uint32 l_1649 = l_1647 + uint32(2);
const uint32_4 l_1650 = buffer_1[l_1649];
const uint32 l_1651 = l_1639.i_0;
const uint32 l_1652 = l_1651 & uint32(252645135);
const uint32 l_1653 = l_1651 >> uint32(4);
const uint32 l_1654 = l_1653 & uint32(252645135);
const uint32 l_1655 = l_1648.i_0;
const int8_4 l_1656 = reinterpret_cast<int8_4 const&>(l_1652);
const int32_4 l_1657 = int32_4{
int32(l_1656.i_0),int32(l_1656.i_1),int32(l_1656.i_2),int32(l_1656.i_3),};
const int8_4 l_1658 = reinterpret_cast<int8_4 const&>(l_1655);
const int32_4 l_1659 = int32_4{
int32(l_1658.i_0),int32(l_1658.i_1),int32(l_1658.i_2),int32(l_1658.i_3),};
#if defined(__gfx1201__)
const int32 l_1660 = __builtin_amdgcn_sudot4(true, l_1652, true, l_1655, 0, false);
#else
const int32 l_1660 = l_1657.i_0 * l_1659.i_0 + l_1657.i_1 * l_1659.i_1 + l_1657.i_2 * l_1659.i_2 + l_1657.i_3 * l_1659.i_3;
#endif
const int32 l_1661 = l_arr_1631[uint32(0)];
const int32 l_1662 = l_1661 + l_1660;
l_arr_1631[uint32(0)] = l_1662;
const uint32 l_1663 = l_1650.i_0;
const int8_4 l_1664 = reinterpret_cast<int8_4 const&>(l_1654);
const int32_4 l_1665 = int32_4{
int32(l_1664.i_0),int32(l_1664.i_1),int32(l_1664.i_2),int32(l_1664.i_3),};
const int8_4 l_1666 = reinterpret_cast<int8_4 const&>(l_1663);
const int32_4 l_1667 = int32_4{
int32(l_1666.i_0),int32(l_1666.i_1),int32(l_1666.i_2),int32(l_1666.i_3),};
#if defined(__gfx1201__)
const int32 l_1668 = __builtin_amdgcn_sudot4(true, l_1654, true, l_1663, 0, false);
#else
const int32 l_1668 = l_1665.i_0 * l_1667.i_0 + l_1665.i_1 * l_1667.i_1 + l_1665.i_2 * l_1667.i_2 + l_1665.i_3 * l_1667.i_3;
#endif
const int32 l_1669 = l_arr_1632[uint32(0)];
const int32 l_1670 = l_1669 + l_1668;
l_arr_1632[uint32(0)] = l_1670;
const uint32 l_1671 = l_1648.i_0;
const uint32 l_1672_bits = uint32(16843009);
const int8_4 l_1672 = reinterpret_cast<int8_4 const&>(l_1672_bits);
const int32_4 l_1673 = int32_4{
int32(l_1672.i_0),int32(l_1672.i_1),int32(l_1672.i_2),int32(l_1672.i_3),};
const int8_4 l_1674 = reinterpret_cast<int8_4 const&>(l_1671);
const int32_4 l_1675 = int32_4{
int32(l_1674.i_0),int32(l_1674.i_1),int32(l_1674.i_2),int32(l_1674.i_3),};
#if defined(__gfx1201__)
const int32 l_1676 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1671, 0, false);
#else
const int32 l_1676 = l_1673.i_0 * l_1675.i_0 + l_1673.i_1 * l_1675.i_1 + l_1673.i_2 * l_1675.i_2 + l_1673.i_3 * l_1675.i_3;
#endif
const int32 l_1677 = l_arr_1633[uint32(0)];
const int32 l_1678 = l_1677 + l_1676;
l_arr_1633[uint32(0)] = l_1678;
const uint32 l_1679 = l_1650.i_0;
const uint32 l_1680_bits = uint32(16843009);
const int8_4 l_1680 = reinterpret_cast<int8_4 const&>(l_1680_bits);
const int32_4 l_1681 = int32_4{
int32(l_1680.i_0),int32(l_1680.i_1),int32(l_1680.i_2),int32(l_1680.i_3),};
const int8_4 l_1682 = reinterpret_cast<int8_4 const&>(l_1679);
const int32_4 l_1683 = int32_4{
int32(l_1682.i_0),int32(l_1682.i_1),int32(l_1682.i_2),int32(l_1682.i_3),};
#if defined(__gfx1201__)
const int32 l_1684 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1679, 0, false);
#else
const int32 l_1684 = l_1681.i_0 * l_1683.i_0 + l_1681.i_1 * l_1683.i_1 + l_1681.i_2 * l_1683.i_2 + l_1681.i_3 * l_1683.i_3;
#endif
const int32 l_1685 = l_arr_1634[uint32(0)];
const int32 l_1686 = l_1685 + l_1684;
l_arr_1634[uint32(0)] = l_1686;
const uint32 l_1687 = l_1639.i_1;
const uint32 l_1688 = l_1687 & uint32(252645135);
const uint32 l_1689 = l_1687 >> uint32(4);
const uint32 l_1690 = l_1689 & uint32(252645135);
const uint32 l_1691 = l_1648.i_1;
const int8_4 l_1692 = reinterpret_cast<int8_4 const&>(l_1688);
const int32_4 l_1693 = int32_4{
int32(l_1692.i_0),int32(l_1692.i_1),int32(l_1692.i_2),int32(l_1692.i_3),};
const int8_4 l_1694 = reinterpret_cast<int8_4 const&>(l_1691);
const int32_4 l_1695 = int32_4{
int32(l_1694.i_0),int32(l_1694.i_1),int32(l_1694.i_2),int32(l_1694.i_3),};
#if defined(__gfx1201__)
const int32 l_1696 = __builtin_amdgcn_sudot4(true, l_1688, true, l_1691, 0, false);
#else
const int32 l_1696 = l_1693.i_0 * l_1695.i_0 + l_1693.i_1 * l_1695.i_1 + l_1693.i_2 * l_1695.i_2 + l_1693.i_3 * l_1695.i_3;
#endif
const int32 l_1697 = l_arr_1631[uint32(0)];
const int32 l_1698 = l_1697 + l_1696;
l_arr_1631[uint32(0)] = l_1698;
const uint32 l_1699 = l_1650.i_1;
const int8_4 l_1700 = reinterpret_cast<int8_4 const&>(l_1690);
const int32_4 l_1701 = int32_4{
int32(l_1700.i_0),int32(l_1700.i_1),int32(l_1700.i_2),int32(l_1700.i_3),};
const int8_4 l_1702 = reinterpret_cast<int8_4 const&>(l_1699);
const int32_4 l_1703 = int32_4{
int32(l_1702.i_0),int32(l_1702.i_1),int32(l_1702.i_2),int32(l_1702.i_3),};
#if defined(__gfx1201__)
const int32 l_1704 = __builtin_amdgcn_sudot4(true, l_1690, true, l_1699, 0, false);
#else
const int32 l_1704 = l_1701.i_0 * l_1703.i_0 + l_1701.i_1 * l_1703.i_1 + l_1701.i_2 * l_1703.i_2 + l_1701.i_3 * l_1703.i_3;
#endif
const int32 l_1705 = l_arr_1632[uint32(0)];
const int32 l_1706 = l_1705 + l_1704;
l_arr_1632[uint32(0)] = l_1706;
const uint32 l_1707 = l_1648.i_1;
const uint32 l_1708_bits = uint32(16843009);
const int8_4 l_1708 = reinterpret_cast<int8_4 const&>(l_1708_bits);
const int32_4 l_1709 = int32_4{
int32(l_1708.i_0),int32(l_1708.i_1),int32(l_1708.i_2),int32(l_1708.i_3),};
const int8_4 l_1710 = reinterpret_cast<int8_4 const&>(l_1707);
const int32_4 l_1711 = int32_4{
int32(l_1710.i_0),int32(l_1710.i_1),int32(l_1710.i_2),int32(l_1710.i_3),};
#if defined(__gfx1201__)
const int32 l_1712 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1707, 0, false);
#else
const int32 l_1712 = l_1709.i_0 * l_1711.i_0 + l_1709.i_1 * l_1711.i_1 + l_1709.i_2 * l_1711.i_2 + l_1709.i_3 * l_1711.i_3;
#endif
const int32 l_1713 = l_arr_1633[uint32(0)];
const int32 l_1714 = l_1713 + l_1712;
l_arr_1633[uint32(0)] = l_1714;
const uint32 l_1715 = l_1650.i_1;
const uint32 l_1716_bits = uint32(16843009);
const int8_4 l_1716 = reinterpret_cast<int8_4 const&>(l_1716_bits);
const int32_4 l_1717 = int32_4{
int32(l_1716.i_0),int32(l_1716.i_1),int32(l_1716.i_2),int32(l_1716.i_3),};
const int8_4 l_1718 = reinterpret_cast<int8_4 const&>(l_1715);
const int32_4 l_1719 = int32_4{
int32(l_1718.i_0),int32(l_1718.i_1),int32(l_1718.i_2),int32(l_1718.i_3),};
#if defined(__gfx1201__)
const int32 l_1720 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1715, 0, false);
#else
const int32 l_1720 = l_1717.i_0 * l_1719.i_0 + l_1717.i_1 * l_1719.i_1 + l_1717.i_2 * l_1719.i_2 + l_1717.i_3 * l_1719.i_3;
#endif
const int32 l_1721 = l_arr_1634[uint32(0)];
const int32 l_1722 = l_1721 + l_1720;
l_arr_1634[uint32(0)] = l_1722;
const uint32 l_1723 = l_1639.i_2;
const uint32 l_1724 = l_1723 & uint32(252645135);
const uint32 l_1725 = l_1723 >> uint32(4);
const uint32 l_1726 = l_1725 & uint32(252645135);
const uint32 l_1727 = l_1648.i_2;
const int8_4 l_1728 = reinterpret_cast<int8_4 const&>(l_1724);
const int32_4 l_1729 = int32_4{
int32(l_1728.i_0),int32(l_1728.i_1),int32(l_1728.i_2),int32(l_1728.i_3),};
const int8_4 l_1730 = reinterpret_cast<int8_4 const&>(l_1727);
const int32_4 l_1731 = int32_4{
int32(l_1730.i_0),int32(l_1730.i_1),int32(l_1730.i_2),int32(l_1730.i_3),};
#if defined(__gfx1201__)
const int32 l_1732 = __builtin_amdgcn_sudot4(true, l_1724, true, l_1727, 0, false);
#else
const int32 l_1732 = l_1729.i_0 * l_1731.i_0 + l_1729.i_1 * l_1731.i_1 + l_1729.i_2 * l_1731.i_2 + l_1729.i_3 * l_1731.i_3;
#endif
const int32 l_1733 = l_arr_1631[uint32(0)];
const int32 l_1734 = l_1733 + l_1732;
l_arr_1631[uint32(0)] = l_1734;
const uint32 l_1735 = l_1650.i_2;
const int8_4 l_1736 = reinterpret_cast<int8_4 const&>(l_1726);
const int32_4 l_1737 = int32_4{
int32(l_1736.i_0),int32(l_1736.i_1),int32(l_1736.i_2),int32(l_1736.i_3),};
const int8_4 l_1738 = reinterpret_cast<int8_4 const&>(l_1735);
const int32_4 l_1739 = int32_4{
int32(l_1738.i_0),int32(l_1738.i_1),int32(l_1738.i_2),int32(l_1738.i_3),};
#if defined(__gfx1201__)
const int32 l_1740 = __builtin_amdgcn_sudot4(true, l_1726, true, l_1735, 0, false);
#else
const int32 l_1740 = l_1737.i_0 * l_1739.i_0 + l_1737.i_1 * l_1739.i_1 + l_1737.i_2 * l_1739.i_2 + l_1737.i_3 * l_1739.i_3;
#endif
const int32 l_1741 = l_arr_1632[uint32(0)];
const int32 l_1742 = l_1741 + l_1740;
l_arr_1632[uint32(0)] = l_1742;
const uint32 l_1743 = l_1648.i_2;
const uint32 l_1744_bits = uint32(16843009);
const int8_4 l_1744 = reinterpret_cast<int8_4 const&>(l_1744_bits);
const int32_4 l_1745 = int32_4{
int32(l_1744.i_0),int32(l_1744.i_1),int32(l_1744.i_2),int32(l_1744.i_3),};
const int8_4 l_1746 = reinterpret_cast<int8_4 const&>(l_1743);
const int32_4 l_1747 = int32_4{
int32(l_1746.i_0),int32(l_1746.i_1),int32(l_1746.i_2),int32(l_1746.i_3),};
#if defined(__gfx1201__)
const int32 l_1748 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1743, 0, false);
#else
const int32 l_1748 = l_1745.i_0 * l_1747.i_0 + l_1745.i_1 * l_1747.i_1 + l_1745.i_2 * l_1747.i_2 + l_1745.i_3 * l_1747.i_3;
#endif
const int32 l_1749 = l_arr_1633[uint32(0)];
const int32 l_1750 = l_1749 + l_1748;
l_arr_1633[uint32(0)] = l_1750;
const uint32 l_1751 = l_1650.i_2;
const uint32 l_1752_bits = uint32(16843009);
const int8_4 l_1752 = reinterpret_cast<int8_4 const&>(l_1752_bits);
const int32_4 l_1753 = int32_4{
int32(l_1752.i_0),int32(l_1752.i_1),int32(l_1752.i_2),int32(l_1752.i_3),};
const int8_4 l_1754 = reinterpret_cast<int8_4 const&>(l_1751);
const int32_4 l_1755 = int32_4{
int32(l_1754.i_0),int32(l_1754.i_1),int32(l_1754.i_2),int32(l_1754.i_3),};
#if defined(__gfx1201__)
const int32 l_1756 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1751, 0, false);
#else
const int32 l_1756 = l_1753.i_0 * l_1755.i_0 + l_1753.i_1 * l_1755.i_1 + l_1753.i_2 * l_1755.i_2 + l_1753.i_3 * l_1755.i_3;
#endif
const int32 l_1757 = l_arr_1634[uint32(0)];
const int32 l_1758 = l_1757 + l_1756;
l_arr_1634[uint32(0)] = l_1758;
const uint32 l_1759 = l_1639.i_3;
const uint32 l_1760 = l_1759 & uint32(252645135);
const uint32 l_1761 = l_1759 >> uint32(4);
const uint32 l_1762 = l_1761 & uint32(252645135);
const uint32 l_1763 = l_1648.i_3;
const int8_4 l_1764 = reinterpret_cast<int8_4 const&>(l_1760);
const int32_4 l_1765 = int32_4{
int32(l_1764.i_0),int32(l_1764.i_1),int32(l_1764.i_2),int32(l_1764.i_3),};
const int8_4 l_1766 = reinterpret_cast<int8_4 const&>(l_1763);
const int32_4 l_1767 = int32_4{
int32(l_1766.i_0),int32(l_1766.i_1),int32(l_1766.i_2),int32(l_1766.i_3),};
#if defined(__gfx1201__)
const int32 l_1768 = __builtin_amdgcn_sudot4(true, l_1760, true, l_1763, 0, false);
#else
const int32 l_1768 = l_1765.i_0 * l_1767.i_0 + l_1765.i_1 * l_1767.i_1 + l_1765.i_2 * l_1767.i_2 + l_1765.i_3 * l_1767.i_3;
#endif
const int32 l_1769 = l_arr_1631[uint32(0)];
const int32 l_1770 = l_1769 + l_1768;
l_arr_1631[uint32(0)] = l_1770;
const uint32 l_1771 = l_1650.i_3;
const int8_4 l_1772 = reinterpret_cast<int8_4 const&>(l_1762);
const int32_4 l_1773 = int32_4{
int32(l_1772.i_0),int32(l_1772.i_1),int32(l_1772.i_2),int32(l_1772.i_3),};
const int8_4 l_1774 = reinterpret_cast<int8_4 const&>(l_1771);
const int32_4 l_1775 = int32_4{
int32(l_1774.i_0),int32(l_1774.i_1),int32(l_1774.i_2),int32(l_1774.i_3),};
#if defined(__gfx1201__)
const int32 l_1776 = __builtin_amdgcn_sudot4(true, l_1762, true, l_1771, 0, false);
#else
const int32 l_1776 = l_1773.i_0 * l_1775.i_0 + l_1773.i_1 * l_1775.i_1 + l_1773.i_2 * l_1775.i_2 + l_1773.i_3 * l_1775.i_3;
#endif
const int32 l_1777 = l_arr_1632[uint32(0)];
const int32 l_1778 = l_1777 + l_1776;
l_arr_1632[uint32(0)] = l_1778;
const uint32 l_1779 = l_1648.i_3;
const uint32 l_1780_bits = uint32(16843009);
const int8_4 l_1780 = reinterpret_cast<int8_4 const&>(l_1780_bits);
const int32_4 l_1781 = int32_4{
int32(l_1780.i_0),int32(l_1780.i_1),int32(l_1780.i_2),int32(l_1780.i_3),};
const int8_4 l_1782 = reinterpret_cast<int8_4 const&>(l_1779);
const int32_4 l_1783 = int32_4{
int32(l_1782.i_0),int32(l_1782.i_1),int32(l_1782.i_2),int32(l_1782.i_3),};
#if defined(__gfx1201__)
const int32 l_1784 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1779, 0, false);
#else
const int32 l_1784 = l_1781.i_0 * l_1783.i_0 + l_1781.i_1 * l_1783.i_1 + l_1781.i_2 * l_1783.i_2 + l_1781.i_3 * l_1783.i_3;
#endif
const int32 l_1785 = l_arr_1633[uint32(0)];
const int32 l_1786 = l_1785 + l_1784;
l_arr_1633[uint32(0)] = l_1786;
const uint32 l_1787 = l_1650.i_3;
const uint32 l_1788_bits = uint32(16843009);
const int8_4 l_1788 = reinterpret_cast<int8_4 const&>(l_1788_bits);
const int32_4 l_1789 = int32_4{
int32(l_1788.i_0),int32(l_1788.i_1),int32(l_1788.i_2),int32(l_1788.i_3),};
const int8_4 l_1790 = reinterpret_cast<int8_4 const&>(l_1787);
const int32_4 l_1791 = int32_4{
int32(l_1790.i_0),int32(l_1790.i_1),int32(l_1790.i_2),int32(l_1790.i_3),};
#if defined(__gfx1201__)
const int32 l_1792 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1787, 0, false);
#else
const int32 l_1792 = l_1789.i_0 * l_1791.i_0 + l_1789.i_1 * l_1791.i_1 + l_1789.i_2 * l_1791.i_2 + l_1789.i_3 * l_1791.i_3;
#endif
const int32 l_1793 = l_arr_1634[uint32(0)];
const int32 l_1794 = l_1793 + l_1792;
l_arr_1634[uint32(0)] = l_1794;
const uint32 l_1795 = l_1593 + uint32(1);
const uint32 l_1796 = l_1590 * uint32(2);
const uint32 l_1797 = l_1795 + l_1796;
const uint32 l_1798 = l_1797 + uint32(1);
const uint32_4 l_1799 = buffer_0[l_1798];
const uint32 l_1800 = l_arr_1581[uint32(0)];
const uint32 l_1801 = l_1 / uint32(16);
const uint32 l_1802 = l_1800 * l_1801;
const uint32 l_1803 = l_1589 * uint32(16);
const uint32 l_1804 = l_1802 + l_1803;
const uint32 l_1805 = l_1590 * uint32(4);
const uint32 l_1806 = l_1804 + l_1805;
const uint32 l_1807 = l_1806 + uint32(1);
const uint32_4 l_1808 = buffer_1[l_1807];
const uint32 l_1809 = l_1807 + uint32(2);
const uint32_4 l_1810 = buffer_1[l_1809];
const uint32 l_1811 = l_1799.i_0;
const uint32 l_1812 = l_1811 & uint32(252645135);
const uint32 l_1813 = l_1811 >> uint32(4);
const uint32 l_1814 = l_1813 & uint32(252645135);
const uint32 l_1815 = l_1808.i_0;
const int8_4 l_1816 = reinterpret_cast<int8_4 const&>(l_1812);
const int32_4 l_1817 = int32_4{
int32(l_1816.i_0),int32(l_1816.i_1),int32(l_1816.i_2),int32(l_1816.i_3),};
const int8_4 l_1818 = reinterpret_cast<int8_4 const&>(l_1815);
const int32_4 l_1819 = int32_4{
int32(l_1818.i_0),int32(l_1818.i_1),int32(l_1818.i_2),int32(l_1818.i_3),};
#if defined(__gfx1201__)
const int32 l_1820 = __builtin_amdgcn_sudot4(true, l_1812, true, l_1815, 0, false);
#else
const int32 l_1820 = l_1817.i_0 * l_1819.i_0 + l_1817.i_1 * l_1819.i_1 + l_1817.i_2 * l_1819.i_2 + l_1817.i_3 * l_1819.i_3;
#endif
const int32 l_1821 = l_arr_1631[uint32(0)];
const int32 l_1822 = l_1821 + l_1820;
l_arr_1631[uint32(0)] = l_1822;
const uint32 l_1823 = l_1810.i_0;
const int8_4 l_1824 = reinterpret_cast<int8_4 const&>(l_1814);
const int32_4 l_1825 = int32_4{
int32(l_1824.i_0),int32(l_1824.i_1),int32(l_1824.i_2),int32(l_1824.i_3),};
const int8_4 l_1826 = reinterpret_cast<int8_4 const&>(l_1823);
const int32_4 l_1827 = int32_4{
int32(l_1826.i_0),int32(l_1826.i_1),int32(l_1826.i_2),int32(l_1826.i_3),};
#if defined(__gfx1201__)
const int32 l_1828 = __builtin_amdgcn_sudot4(true, l_1814, true, l_1823, 0, false);
#else
const int32 l_1828 = l_1825.i_0 * l_1827.i_0 + l_1825.i_1 * l_1827.i_1 + l_1825.i_2 * l_1827.i_2 + l_1825.i_3 * l_1827.i_3;
#endif
const int32 l_1829 = l_arr_1632[uint32(0)];
const int32 l_1830 = l_1829 + l_1828;
l_arr_1632[uint32(0)] = l_1830;
const uint32 l_1831 = l_1808.i_0;
const uint32 l_1832_bits = uint32(16843009);
const int8_4 l_1832 = reinterpret_cast<int8_4 const&>(l_1832_bits);
const int32_4 l_1833 = int32_4{
int32(l_1832.i_0),int32(l_1832.i_1),int32(l_1832.i_2),int32(l_1832.i_3),};
const int8_4 l_1834 = reinterpret_cast<int8_4 const&>(l_1831);
const int32_4 l_1835 = int32_4{
int32(l_1834.i_0),int32(l_1834.i_1),int32(l_1834.i_2),int32(l_1834.i_3),};
#if defined(__gfx1201__)
const int32 l_1836 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1831, 0, false);
#else
const int32 l_1836 = l_1833.i_0 * l_1835.i_0 + l_1833.i_1 * l_1835.i_1 + l_1833.i_2 * l_1835.i_2 + l_1833.i_3 * l_1835.i_3;
#endif
const int32 l_1837 = l_arr_1633[uint32(0)];
const int32 l_1838 = l_1837 + l_1836;
l_arr_1633[uint32(0)] = l_1838;
const uint32 l_1839 = l_1810.i_0;
const uint32 l_1840_bits = uint32(16843009);
const int8_4 l_1840 = reinterpret_cast<int8_4 const&>(l_1840_bits);
const int32_4 l_1841 = int32_4{
int32(l_1840.i_0),int32(l_1840.i_1),int32(l_1840.i_2),int32(l_1840.i_3),};
const int8_4 l_1842 = reinterpret_cast<int8_4 const&>(l_1839);
const int32_4 l_1843 = int32_4{
int32(l_1842.i_0),int32(l_1842.i_1),int32(l_1842.i_2),int32(l_1842.i_3),};
#if defined(__gfx1201__)
const int32 l_1844 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1839, 0, false);
#else
const int32 l_1844 = l_1841.i_0 * l_1843.i_0 + l_1841.i_1 * l_1843.i_1 + l_1841.i_2 * l_1843.i_2 + l_1841.i_3 * l_1843.i_3;
#endif
const int32 l_1845 = l_arr_1634[uint32(0)];
const int32 l_1846 = l_1845 + l_1844;
l_arr_1634[uint32(0)] = l_1846;
const uint32 l_1847 = l_1799.i_1;
const uint32 l_1848 = l_1847 & uint32(252645135);
const uint32 l_1849 = l_1847 >> uint32(4);
const uint32 l_1850 = l_1849 & uint32(252645135);
const uint32 l_1851 = l_1808.i_1;
const int8_4 l_1852 = reinterpret_cast<int8_4 const&>(l_1848);
const int32_4 l_1853 = int32_4{
int32(l_1852.i_0),int32(l_1852.i_1),int32(l_1852.i_2),int32(l_1852.i_3),};
const int8_4 l_1854 = reinterpret_cast<int8_4 const&>(l_1851);
const int32_4 l_1855 = int32_4{
int32(l_1854.i_0),int32(l_1854.i_1),int32(l_1854.i_2),int32(l_1854.i_3),};
#if defined(__gfx1201__)
const int32 l_1856 = __builtin_amdgcn_sudot4(true, l_1848, true, l_1851, 0, false);
#else
const int32 l_1856 = l_1853.i_0 * l_1855.i_0 + l_1853.i_1 * l_1855.i_1 + l_1853.i_2 * l_1855.i_2 + l_1853.i_3 * l_1855.i_3;
#endif
const int32 l_1857 = l_arr_1631[uint32(0)];
const int32 l_1858 = l_1857 + l_1856;
l_arr_1631[uint32(0)] = l_1858;
const uint32 l_1859 = l_1810.i_1;
const int8_4 l_1860 = reinterpret_cast<int8_4 const&>(l_1850);
const int32_4 l_1861 = int32_4{
int32(l_1860.i_0),int32(l_1860.i_1),int32(l_1860.i_2),int32(l_1860.i_3),};
const int8_4 l_1862 = reinterpret_cast<int8_4 const&>(l_1859);
const int32_4 l_1863 = int32_4{
int32(l_1862.i_0),int32(l_1862.i_1),int32(l_1862.i_2),int32(l_1862.i_3),};
#if defined(__gfx1201__)
const int32 l_1864 = __builtin_amdgcn_sudot4(true, l_1850, true, l_1859, 0, false);
#else
const int32 l_1864 = l_1861.i_0 * l_1863.i_0 + l_1861.i_1 * l_1863.i_1 + l_1861.i_2 * l_1863.i_2 + l_1861.i_3 * l_1863.i_3;
#endif
const int32 l_1865 = l_arr_1632[uint32(0)];
const int32 l_1866 = l_1865 + l_1864;
l_arr_1632[uint32(0)] = l_1866;
const uint32 l_1867 = l_1808.i_1;
const uint32 l_1868_bits = uint32(16843009);
const int8_4 l_1868 = reinterpret_cast<int8_4 const&>(l_1868_bits);
const int32_4 l_1869 = int32_4{
int32(l_1868.i_0),int32(l_1868.i_1),int32(l_1868.i_2),int32(l_1868.i_3),};
const int8_4 l_1870 = reinterpret_cast<int8_4 const&>(l_1867);
const int32_4 l_1871 = int32_4{
int32(l_1870.i_0),int32(l_1870.i_1),int32(l_1870.i_2),int32(l_1870.i_3),};
#if defined(__gfx1201__)
const int32 l_1872 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1867, 0, false);
#else
const int32 l_1872 = l_1869.i_0 * l_1871.i_0 + l_1869.i_1 * l_1871.i_1 + l_1869.i_2 * l_1871.i_2 + l_1869.i_3 * l_1871.i_3;
#endif
const int32 l_1873 = l_arr_1633[uint32(0)];
const int32 l_1874 = l_1873 + l_1872;
l_arr_1633[uint32(0)] = l_1874;
const uint32 l_1875 = l_1810.i_1;
const uint32 l_1876_bits = uint32(16843009);
const int8_4 l_1876 = reinterpret_cast<int8_4 const&>(l_1876_bits);
const int32_4 l_1877 = int32_4{
int32(l_1876.i_0),int32(l_1876.i_1),int32(l_1876.i_2),int32(l_1876.i_3),};
const int8_4 l_1878 = reinterpret_cast<int8_4 const&>(l_1875);
const int32_4 l_1879 = int32_4{
int32(l_1878.i_0),int32(l_1878.i_1),int32(l_1878.i_2),int32(l_1878.i_3),};
#if defined(__gfx1201__)
const int32 l_1880 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1875, 0, false);
#else
const int32 l_1880 = l_1877.i_0 * l_1879.i_0 + l_1877.i_1 * l_1879.i_1 + l_1877.i_2 * l_1879.i_2 + l_1877.i_3 * l_1879.i_3;
#endif
const int32 l_1881 = l_arr_1634[uint32(0)];
const int32 l_1882 = l_1881 + l_1880;
l_arr_1634[uint32(0)] = l_1882;
const uint32 l_1883 = l_1799.i_2;
const uint32 l_1884 = l_1883 & uint32(252645135);
const uint32 l_1885 = l_1883 >> uint32(4);
const uint32 l_1886 = l_1885 & uint32(252645135);
const uint32 l_1887 = l_1808.i_2;
const int8_4 l_1888 = reinterpret_cast<int8_4 const&>(l_1884);
const int32_4 l_1889 = int32_4{
int32(l_1888.i_0),int32(l_1888.i_1),int32(l_1888.i_2),int32(l_1888.i_3),};
const int8_4 l_1890 = reinterpret_cast<int8_4 const&>(l_1887);
const int32_4 l_1891 = int32_4{
int32(l_1890.i_0),int32(l_1890.i_1),int32(l_1890.i_2),int32(l_1890.i_3),};
#if defined(__gfx1201__)
const int32 l_1892 = __builtin_amdgcn_sudot4(true, l_1884, true, l_1887, 0, false);
#else
const int32 l_1892 = l_1889.i_0 * l_1891.i_0 + l_1889.i_1 * l_1891.i_1 + l_1889.i_2 * l_1891.i_2 + l_1889.i_3 * l_1891.i_3;
#endif
const int32 l_1893 = l_arr_1631[uint32(0)];
const int32 l_1894 = l_1893 + l_1892;
l_arr_1631[uint32(0)] = l_1894;
const uint32 l_1895 = l_1810.i_2;
const int8_4 l_1896 = reinterpret_cast<int8_4 const&>(l_1886);
const int32_4 l_1897 = int32_4{
int32(l_1896.i_0),int32(l_1896.i_1),int32(l_1896.i_2),int32(l_1896.i_3),};
const int8_4 l_1898 = reinterpret_cast<int8_4 const&>(l_1895);
const int32_4 l_1899 = int32_4{
int32(l_1898.i_0),int32(l_1898.i_1),int32(l_1898.i_2),int32(l_1898.i_3),};
#if defined(__gfx1201__)
const int32 l_1900 = __builtin_amdgcn_sudot4(true, l_1886, true, l_1895, 0, false);
#else
const int32 l_1900 = l_1897.i_0 * l_1899.i_0 + l_1897.i_1 * l_1899.i_1 + l_1897.i_2 * l_1899.i_2 + l_1897.i_3 * l_1899.i_3;
#endif
const int32 l_1901 = l_arr_1632[uint32(0)];
const int32 l_1902 = l_1901 + l_1900;
l_arr_1632[uint32(0)] = l_1902;
const uint32 l_1903 = l_1808.i_2;
const uint32 l_1904_bits = uint32(16843009);
const int8_4 l_1904 = reinterpret_cast<int8_4 const&>(l_1904_bits);
const int32_4 l_1905 = int32_4{
int32(l_1904.i_0),int32(l_1904.i_1),int32(l_1904.i_2),int32(l_1904.i_3),};
const int8_4 l_1906 = reinterpret_cast<int8_4 const&>(l_1903);
const int32_4 l_1907 = int32_4{
int32(l_1906.i_0),int32(l_1906.i_1),int32(l_1906.i_2),int32(l_1906.i_3),};
#if defined(__gfx1201__)
const int32 l_1908 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1903, 0, false);
#else
const int32 l_1908 = l_1905.i_0 * l_1907.i_0 + l_1905.i_1 * l_1907.i_1 + l_1905.i_2 * l_1907.i_2 + l_1905.i_3 * l_1907.i_3;
#endif
const int32 l_1909 = l_arr_1633[uint32(0)];
const int32 l_1910 = l_1909 + l_1908;
l_arr_1633[uint32(0)] = l_1910;
const uint32 l_1911 = l_1810.i_2;
const uint32 l_1912_bits = uint32(16843009);
const int8_4 l_1912 = reinterpret_cast<int8_4 const&>(l_1912_bits);
const int32_4 l_1913 = int32_4{
int32(l_1912.i_0),int32(l_1912.i_1),int32(l_1912.i_2),int32(l_1912.i_3),};
const int8_4 l_1914 = reinterpret_cast<int8_4 const&>(l_1911);
const int32_4 l_1915 = int32_4{
int32(l_1914.i_0),int32(l_1914.i_1),int32(l_1914.i_2),int32(l_1914.i_3),};
#if defined(__gfx1201__)
const int32 l_1916 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1911, 0, false);
#else
const int32 l_1916 = l_1913.i_0 * l_1915.i_0 + l_1913.i_1 * l_1915.i_1 + l_1913.i_2 * l_1915.i_2 + l_1913.i_3 * l_1915.i_3;
#endif
const int32 l_1917 = l_arr_1634[uint32(0)];
const int32 l_1918 = l_1917 + l_1916;
l_arr_1634[uint32(0)] = l_1918;
const uint32 l_1919 = l_1799.i_3;
const uint32 l_1920 = l_1919 & uint32(252645135);
const uint32 l_1921 = l_1919 >> uint32(4);
const uint32 l_1922 = l_1921 & uint32(252645135);
const uint32 l_1923 = l_1808.i_3;
const int8_4 l_1924 = reinterpret_cast<int8_4 const&>(l_1920);
const int32_4 l_1925 = int32_4{
int32(l_1924.i_0),int32(l_1924.i_1),int32(l_1924.i_2),int32(l_1924.i_3),};
const int8_4 l_1926 = reinterpret_cast<int8_4 const&>(l_1923);
const int32_4 l_1927 = int32_4{
int32(l_1926.i_0),int32(l_1926.i_1),int32(l_1926.i_2),int32(l_1926.i_3),};
#if defined(__gfx1201__)
const int32 l_1928 = __builtin_amdgcn_sudot4(true, l_1920, true, l_1923, 0, false);
#else
const int32 l_1928 = l_1925.i_0 * l_1927.i_0 + l_1925.i_1 * l_1927.i_1 + l_1925.i_2 * l_1927.i_2 + l_1925.i_3 * l_1927.i_3;
#endif
const int32 l_1929 = l_arr_1631[uint32(0)];
const int32 l_1930 = l_1929 + l_1928;
l_arr_1631[uint32(0)] = l_1930;
const uint32 l_1931 = l_1810.i_3;
const int8_4 l_1932 = reinterpret_cast<int8_4 const&>(l_1922);
const int32_4 l_1933 = int32_4{
int32(l_1932.i_0),int32(l_1932.i_1),int32(l_1932.i_2),int32(l_1932.i_3),};
const int8_4 l_1934 = reinterpret_cast<int8_4 const&>(l_1931);
const int32_4 l_1935 = int32_4{
int32(l_1934.i_0),int32(l_1934.i_1),int32(l_1934.i_2),int32(l_1934.i_3),};
#if defined(__gfx1201__)
const int32 l_1936 = __builtin_amdgcn_sudot4(true, l_1922, true, l_1931, 0, false);
#else
const int32 l_1936 = l_1933.i_0 * l_1935.i_0 + l_1933.i_1 * l_1935.i_1 + l_1933.i_2 * l_1935.i_2 + l_1933.i_3 * l_1935.i_3;
#endif
const int32 l_1937 = l_arr_1632[uint32(0)];
const int32 l_1938 = l_1937 + l_1936;
l_arr_1632[uint32(0)] = l_1938;
const uint32 l_1939 = l_1808.i_3;
const uint32 l_1940_bits = uint32(16843009);
const int8_4 l_1940 = reinterpret_cast<int8_4 const&>(l_1940_bits);
const int32_4 l_1941 = int32_4{
int32(l_1940.i_0),int32(l_1940.i_1),int32(l_1940.i_2),int32(l_1940.i_3),};
const int8_4 l_1942 = reinterpret_cast<int8_4 const&>(l_1939);
const int32_4 l_1943 = int32_4{
int32(l_1942.i_0),int32(l_1942.i_1),int32(l_1942.i_2),int32(l_1942.i_3),};
#if defined(__gfx1201__)
const int32 l_1944 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1939, 0, false);
#else
const int32 l_1944 = l_1941.i_0 * l_1943.i_0 + l_1941.i_1 * l_1943.i_1 + l_1941.i_2 * l_1943.i_2 + l_1941.i_3 * l_1943.i_3;
#endif
const int32 l_1945 = l_arr_1633[uint32(0)];
const int32 l_1946 = l_1945 + l_1944;
l_arr_1633[uint32(0)] = l_1946;
const uint32 l_1947 = l_1810.i_3;
const uint32 l_1948_bits = uint32(16843009);
const int8_4 l_1948 = reinterpret_cast<int8_4 const&>(l_1948_bits);
const int32_4 l_1949 = int32_4{
int32(l_1948.i_0),int32(l_1948.i_1),int32(l_1948.i_2),int32(l_1948.i_3),};
const int8_4 l_1950 = reinterpret_cast<int8_4 const&>(l_1947);
const int32_4 l_1951 = int32_4{
int32(l_1950.i_0),int32(l_1950.i_1),int32(l_1950.i_2),int32(l_1950.i_3),};
#if defined(__gfx1201__)
const int32 l_1952 = __builtin_amdgcn_sudot4(true, uint32(16843009), true, l_1947, 0, false);
#else
const int32 l_1952 = l_1949.i_0 * l_1951.i_0 + l_1949.i_1 * l_1951.i_1 + l_1949.i_2 * l_1951.i_2 + l_1949.i_3 * l_1951.i_3;
#endif
const int32 l_1953 = l_arr_1634[uint32(0)];
const int32 l_1954 = l_1953 + l_1952;
l_arr_1634[uint32(0)] = l_1954;
const uint32 l_1955 = l_arr_1581[uint32(0)];
const uint32 l_1956 = l_1 / uint32(32);
const uint32 l_1957 = l_1955 * l_1956;
const uint32 l_1958 = l_1589 * uint32(8);
const uint32 l_1959 = l_1957 + l_1958;
const uint32 l_1960 = l_1590 * uint32(2);
const uint32 l_1961 = l_1959 + l_1960;
const __half l_1962 = buffer_2[l_1961];
const float l_1963 = float(l_1962);
const uint32 l_1964 = l_1961 + uint32(1);
const __half l_1965 = buffer_2[l_1964];
const float l_1966 = float(l_1965);
const int32 l_1967 = l_arr_1631[uint32(0)];
const int32 l_1968 = l_1967 * l_1622;
const float l_1969 = float(l_1968);
const int32 l_1970 = l_arr_1632[uint32(0)];
const int32 l_1971 = l_1970 * l_1625;
const float l_1972 = float(l_1971);
const float l_1973 = l_1966 * l_1972;
const float l_1974 = fma(l_1963, l_1969, l_1973);
const int32 l_1975 = l_arr_1633[uint32(0)];
const int32 l_1976 = l_1975 * l_1627;
const float l_1977 = float(l_1976);
const int32 l_1978 = l_arr_1634[uint32(0)];
const int32 l_1979 = l_1978 * l_1630;
const float l_1980 = float(l_1979);
const float l_1981 = l_1966 * l_1980;
const float l_1982 = fma(l_1963, l_1977, l_1981);
const float l_1983 = l_1600 * l_1982;
const float l_1984 = -l_1983;
const float l_1985 = fma(l_1598, l_1974, l_1984);
const float l_1986 = l_arr_1582[uint32(0)];
const float l_1987 = l_1986 + l_1985;
l_arr_1582[uint32(0)] = l_1987;
l_mut_67 = l_mut_67 + uint32(32);
}
const float l_1988 = l_arr_1582[uint32(0)];
const float l_1989 = { __shfl_down(l_1988, uint32(16)) };
const float l_1990 = l_arr_1582[uint32(0)];
const float l_1991 = l_1990 + l_1989;
l_arr_1582[uint32(0)] = l_1991;
const float l_1992 = l_arr_1582[uint32(0)];
const float l_1993 = { __shfl_down(l_1992, uint32(8)) };
const float l_1994 = l_arr_1582[uint32(0)];
const float l_1995 = l_1994 + l_1993;
l_arr_1582[uint32(0)] = l_1995;
const float l_1996 = l_arr_1582[uint32(0)];
const float l_1997 = { __shfl_down(l_1996, uint32(4)) };
const float l_1998 = l_arr_1582[uint32(0)];
const float l_1999 = l_1998 + l_1997;
l_arr_1582[uint32(0)] = l_1999;
const float l_2000 = l_arr_1582[uint32(0)];
const float l_2001 = { __shfl_down(l_2000, uint32(2)) };
const float l_2002 = l_arr_1582[uint32(0)];
const float l_2003 = l_2002 + l_2001;
l_arr_1582[uint32(0)] = l_2003;
const float l_2004 = l_arr_1582[uint32(0)];
const float l_2005 = { __shfl_down(l_2004, uint32(1)) };
const float l_2006 = l_arr_1582[uint32(0)];
const float l_2007 = l_2006 + l_2005;
l_arr_1582[uint32(0)] = l_2007;
const bool l_2008 = l_13 == uint32(0);
if (l_2008) {
const uint32 l_2009 = l_arr_1581[uint32(0)];
const uint32 l_2010 = l_2009 * l_0;
const uint32 l_2011 = l_2010 + l_7;
const float l_2012 = l_arr_1582[uint32(0)];
const __half l_2013 = __half(l_2012);
buffer_9[l_2011] = l_2013;
}
l_mut_22 = l_mut_22 + uint32(1);
}
}

}
