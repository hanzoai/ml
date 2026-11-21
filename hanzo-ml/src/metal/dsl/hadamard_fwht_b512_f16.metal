#include <metal_stdlib>
using namespace metal;

struct info_st {
    uint static_meta[8];
};

[[kernel]]
void hadamard_fwht_f_(
    const device half * buffer_0 [[buffer(0)]],
    const device float * buffer_1 [[buffer(1)]],
    device half * buffer_2 [[buffer(2)]],
    const device uint * buffer_3 [[buffer(3)]],
    constant info_st& info [[buffer(4)]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint3 thread_pos_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_pos_in_grid [[threadgroup_position_in_grid]],
    uint simd_size [[threads_per_simdgroup]]
) {
uint threads_per_simdgroup_checked = min(simd_size, threads_per_threadgroup.x * threads_per_threadgroup.y * threads_per_threadgroup.z);
threadgroup uchar dynamic_shared_mem[2048];
// Shared array size: 512, 2048 bytes
threadgroup float* shared_memory_4 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[0]);
float l_arr_5[2];

float l_arr_6[2];

uint l_mut_7;
float l_mut_17;
float l_mut_25;
const uint l_0 = buffer_3[uint(1)];
const uint l_1 = buffer_3[uint(2)];
const uint l_2 = buffer_3[uint(3)];
const uint l_3 = buffer_3[uint(4)];
l_mut_7 = threadgroup_pos_in_grid.x;
while (true) {
const bool l_8 = l_mut_7 < l_3;
const bool l_9 = !l_8;
if (l_9) {
break;}
const uint l_10 = l_mut_7 * uint(512);
const uint l_11 = l_mut_7 % l_0;
const uint l_12 = l_11 * uint(512);
const uint l_13 = uint(0) + thread_pos_in_threadgroup.x;
const uint l_14 = l_10 + l_13;
const half l_15 = buffer_0[l_14];
const float l_16 = float(l_15);
l_mut_17 = l_16;
const bool l_18 = l_2 == uint(0);
const bool l_19 = l_1 != uint(0);
const bool l_20 = l_18 && l_19;
if (l_20) {
const uint l_21 = l_12 + l_13;
const float l_22 = buffer_1[l_21];
l_mut_17 = l_mut_17 * l_22;
}
const float l_23 = { simd_shuffle_xor(l_mut_17, uint(1)) };
const float l_24 = l_mut_17 + l_23;
l_mut_25 = l_24;
const uint l_26 = thread_pos_in_threadgroup.x & uint(1);
const bool l_27 = l_26 != uint(0);
if (l_27) {
const float l_28 = l_23 - l_mut_17;
l_mut_25 = l_28;
}
l_mut_17 = l_mut_25;
const float l_29 = { simd_shuffle_xor(l_mut_17, uint(2)) };
const float l_30 = l_mut_17 + l_29;
l_mut_25 = l_30;
const uint l_31 = thread_pos_in_threadgroup.x & uint(2);
const bool l_32 = l_31 != uint(0);
if (l_32) {
const float l_33 = l_29 - l_mut_17;
l_mut_25 = l_33;
}
l_mut_17 = l_mut_25;
const float l_34 = { simd_shuffle_xor(l_mut_17, uint(4)) };
const float l_35 = l_mut_17 + l_34;
l_mut_25 = l_35;
const uint l_36 = thread_pos_in_threadgroup.x & uint(4);
const bool l_37 = l_36 != uint(0);
if (l_37) {
const float l_38 = l_34 - l_mut_17;
l_mut_25 = l_38;
}
l_mut_17 = l_mut_25;
const float l_39 = { simd_shuffle_xor(l_mut_17, uint(8)) };
const float l_40 = l_mut_17 + l_39;
l_mut_25 = l_40;
const uint l_41 = thread_pos_in_threadgroup.x & uint(8);
const bool l_42 = l_41 != uint(0);
if (l_42) {
const float l_43 = l_39 - l_mut_17;
l_mut_25 = l_43;
}
l_mut_17 = l_mut_25;
const float l_44 = { simd_shuffle_xor(l_mut_17, uint(16)) };
const float l_45 = l_mut_17 + l_44;
l_mut_25 = l_45;
const uint l_46 = thread_pos_in_threadgroup.x & uint(16);
const bool l_47 = l_46 != uint(0);
if (l_47) {
const float l_48 = l_44 - l_mut_17;
l_mut_25 = l_48;
}
l_mut_17 = l_mut_25;
l_arr_5[uint(0)] = l_mut_17;
const uint l_49 = uint(256) + thread_pos_in_threadgroup.x;
const uint l_50 = l_10 + l_49;
const half l_51 = buffer_0[l_50];
const float l_52 = float(l_51);
l_mut_25 = l_52;
const bool l_53 = l_2 == uint(0);
const bool l_54 = l_1 != uint(0);
const bool l_55 = l_53 && l_54;
if (l_55) {
const uint l_56 = l_12 + l_49;
const float l_57 = buffer_1[l_56];
l_mut_25 = l_mut_25 * l_57;
}
const float l_58 = { simd_shuffle_xor(l_mut_25, uint(1)) };
const float l_59 = l_mut_25 + l_58;
l_mut_17 = l_59;
const uint l_60 = thread_pos_in_threadgroup.x & uint(1);
const bool l_61 = l_60 != uint(0);
if (l_61) {
const float l_62 = l_58 - l_mut_25;
l_mut_17 = l_62;
}
l_mut_25 = l_mut_17;
const float l_63 = { simd_shuffle_xor(l_mut_25, uint(2)) };
const float l_64 = l_mut_25 + l_63;
l_mut_17 = l_64;
const uint l_65 = thread_pos_in_threadgroup.x & uint(2);
const bool l_66 = l_65 != uint(0);
if (l_66) {
const float l_67 = l_63 - l_mut_25;
l_mut_17 = l_67;
}
l_mut_25 = l_mut_17;
const float l_68 = { simd_shuffle_xor(l_mut_25, uint(4)) };
const float l_69 = l_mut_25 + l_68;
l_mut_17 = l_69;
const uint l_70 = thread_pos_in_threadgroup.x & uint(4);
const bool l_71 = l_70 != uint(0);
if (l_71) {
const float l_72 = l_68 - l_mut_25;
l_mut_17 = l_72;
}
l_mut_25 = l_mut_17;
const float l_73 = { simd_shuffle_xor(l_mut_25, uint(8)) };
const float l_74 = l_mut_25 + l_73;
l_mut_17 = l_74;
const uint l_75 = thread_pos_in_threadgroup.x & uint(8);
const bool l_76 = l_75 != uint(0);
if (l_76) {
const float l_77 = l_73 - l_mut_25;
l_mut_17 = l_77;
}
l_mut_25 = l_mut_17;
const float l_78 = { simd_shuffle_xor(l_mut_25, uint(16)) };
const float l_79 = l_mut_25 + l_78;
l_mut_17 = l_79;
const uint l_80 = thread_pos_in_threadgroup.x & uint(16);
const bool l_81 = l_80 != uint(0);
if (l_81) {
const float l_82 = l_78 - l_mut_25;
l_mut_17 = l_82;
}
l_mut_25 = l_mut_17;
l_arr_5[uint(1)] = l_mut_25;
const uint l_83 = uint(0) + thread_pos_in_threadgroup.x;
const float l_84 = l_arr_5[uint(0)];
shared_memory_4[l_83] = l_84;
const uint l_85 = uint(256) + thread_pos_in_threadgroup.x;
const float l_86 = l_arr_5[uint(1)];
shared_memory_4[l_85] = l_86;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_87 = thread_pos_in_threadgroup.x ^ uint(32);
const uint l_88 = thread_pos_in_threadgroup.x & uint(32);
const bool l_89 = l_88 != uint(0);
const float l_90 = l_arr_5[uint(0)];
const uint l_91 = uint(0) + l_87;
const float l_92 = shared_memory_4[l_91];
const float l_93 = l_90 + l_92;
l_mut_25 = l_93;
if (l_89) {
const float l_94 = l_92 - l_90;
l_mut_25 = l_94;
}
l_arr_6[uint(0)] = l_mut_25;
const float l_95 = l_arr_5[uint(1)];
const uint l_96 = uint(256) + l_87;
const float l_97 = shared_memory_4[l_96];
const float l_98 = l_95 + l_97;
l_mut_25 = l_98;
if (l_89) {
const float l_99 = l_97 - l_95;
l_mut_25 = l_99;
}
l_arr_6[uint(1)] = l_mut_25;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float l_100 = l_arr_6[uint(0)];
l_arr_5[uint(0)] = l_100;
const uint l_101 = uint(0) + thread_pos_in_threadgroup.x;
shared_memory_4[l_101] = l_100;
const float l_102 = l_arr_6[uint(1)];
l_arr_5[uint(1)] = l_102;
const uint l_103 = uint(256) + thread_pos_in_threadgroup.x;
shared_memory_4[l_103] = l_102;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_104 = thread_pos_in_threadgroup.x ^ uint(64);
const uint l_105 = thread_pos_in_threadgroup.x & uint(64);
const bool l_106 = l_105 != uint(0);
const float l_107 = l_arr_5[uint(0)];
const uint l_108 = uint(0) + l_104;
const float l_109 = shared_memory_4[l_108];
const float l_110 = l_107 + l_109;
l_mut_25 = l_110;
if (l_106) {
const float l_111 = l_109 - l_107;
l_mut_25 = l_111;
}
l_arr_6[uint(0)] = l_mut_25;
const float l_112 = l_arr_5[uint(1)];
const uint l_113 = uint(256) + l_104;
const float l_114 = shared_memory_4[l_113];
const float l_115 = l_112 + l_114;
l_mut_25 = l_115;
if (l_106) {
const float l_116 = l_114 - l_112;
l_mut_25 = l_116;
}
l_arr_6[uint(1)] = l_mut_25;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float l_117 = l_arr_6[uint(0)];
l_arr_5[uint(0)] = l_117;
const uint l_118 = uint(0) + thread_pos_in_threadgroup.x;
shared_memory_4[l_118] = l_117;
const float l_119 = l_arr_6[uint(1)];
l_arr_5[uint(1)] = l_119;
const uint l_120 = uint(256) + thread_pos_in_threadgroup.x;
shared_memory_4[l_120] = l_119;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_121 = thread_pos_in_threadgroup.x ^ uint(128);
const uint l_122 = thread_pos_in_threadgroup.x & uint(128);
const bool l_123 = l_122 != uint(0);
const float l_124 = l_arr_5[uint(0)];
const uint l_125 = uint(0) + l_121;
const float l_126 = shared_memory_4[l_125];
const float l_127 = l_124 + l_126;
l_mut_25 = l_127;
if (l_123) {
const float l_128 = l_126 - l_124;
l_mut_25 = l_128;
}
l_arr_6[uint(0)] = l_mut_25;
const float l_129 = l_arr_5[uint(1)];
const uint l_130 = uint(256) + l_121;
const float l_131 = shared_memory_4[l_130];
const float l_132 = l_129 + l_131;
l_mut_25 = l_132;
if (l_123) {
const float l_133 = l_131 - l_129;
l_mut_25 = l_133;
}
l_arr_6[uint(1)] = l_mut_25;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float l_134 = l_arr_6[uint(0)];
l_arr_5[uint(0)] = l_134;
const float l_135 = l_arr_6[uint(1)];
l_arr_5[uint(1)] = l_135;
const float l_136 = l_arr_5[uint(0)];
const float l_137 = l_arr_5[uint(1)];
const float l_138 = l_136 + l_137;
l_arr_5[uint(0)] = l_138;
const float l_139 = l_136 - l_137;
l_arr_5[uint(1)] = l_139;
const uint l_140 = uint(0) + thread_pos_in_threadgroup.x;
const float l_141 = l_arr_5[uint(0)];
const float l_142 = l_141 * float(0.04419417306780815);
l_mut_25 = l_142;
const bool l_143 = l_2 != uint(0);
const bool l_144 = l_1 != uint(0);
const bool l_145 = l_143 && l_144;
if (l_145) {
const uint l_146 = l_12 + l_140;
const float l_147 = buffer_1[l_146];
l_mut_25 = l_mut_25 * l_147;
}
const uint l_148 = l_10 + l_140;
const half l_149 = half(l_mut_25);
buffer_2[l_148] = l_149;
const uint l_150 = uint(256) + thread_pos_in_threadgroup.x;
const float l_151 = l_arr_5[uint(1)];
const float l_152 = l_151 * float(0.04419417306780815);
l_mut_25 = l_152;
const bool l_153 = l_2 != uint(0);
const bool l_154 = l_1 != uint(0);
const bool l_155 = l_153 && l_154;
if (l_155) {
const uint l_156 = l_12 + l_150;
const float l_157 = buffer_1[l_156];
l_mut_25 = l_mut_25 * l_157;
}
const uint l_158 = l_10 + l_150;
const half l_159 = half(l_mut_25);
buffer_2[l_158] = l_159;
l_mut_7 = l_mut_7 + threadgroups_per_grid.x;
}

}
