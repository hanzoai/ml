#include <metal_stdlib>
using namespace metal;

struct info_st {
    uint static_meta[8];
};

[[kernel]]
void hadamard_fwht_f_f32(
    const device float * buffer_0 [[buffer(0)]],
    const device float * buffer_1 [[buffer(1)]],
    device float * buffer_2 [[buffer(2)]],
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
float l_mut_16;
float l_mut_24;
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
const float l_15 = buffer_0[l_14];
l_mut_16 = l_15;
const bool l_17 = l_2 == uint(0);
const bool l_18 = l_1 != uint(0);
const bool l_19 = l_17 && l_18;
if (l_19) {
const uint l_20 = l_12 + l_13;
const float l_21 = buffer_1[l_20];
l_mut_16 = l_mut_16 * l_21;
}
const float l_22 = { simd_shuffle_xor(l_mut_16, uint(1)) };
const float l_23 = l_mut_16 + l_22;
l_mut_24 = l_23;
const uint l_25 = thread_pos_in_threadgroup.x & uint(1);
const bool l_26 = l_25 != uint(0);
if (l_26) {
const float l_27 = l_22 - l_mut_16;
l_mut_24 = l_27;
}
l_mut_16 = l_mut_24;
const float l_28 = { simd_shuffle_xor(l_mut_16, uint(2)) };
const float l_29 = l_mut_16 + l_28;
l_mut_24 = l_29;
const uint l_30 = thread_pos_in_threadgroup.x & uint(2);
const bool l_31 = l_30 != uint(0);
if (l_31) {
const float l_32 = l_28 - l_mut_16;
l_mut_24 = l_32;
}
l_mut_16 = l_mut_24;
const float l_33 = { simd_shuffle_xor(l_mut_16, uint(4)) };
const float l_34 = l_mut_16 + l_33;
l_mut_24 = l_34;
const uint l_35 = thread_pos_in_threadgroup.x & uint(4);
const bool l_36 = l_35 != uint(0);
if (l_36) {
const float l_37 = l_33 - l_mut_16;
l_mut_24 = l_37;
}
l_mut_16 = l_mut_24;
const float l_38 = { simd_shuffle_xor(l_mut_16, uint(8)) };
const float l_39 = l_mut_16 + l_38;
l_mut_24 = l_39;
const uint l_40 = thread_pos_in_threadgroup.x & uint(8);
const bool l_41 = l_40 != uint(0);
if (l_41) {
const float l_42 = l_38 - l_mut_16;
l_mut_24 = l_42;
}
l_mut_16 = l_mut_24;
const float l_43 = { simd_shuffle_xor(l_mut_16, uint(16)) };
const float l_44 = l_mut_16 + l_43;
l_mut_24 = l_44;
const uint l_45 = thread_pos_in_threadgroup.x & uint(16);
const bool l_46 = l_45 != uint(0);
if (l_46) {
const float l_47 = l_43 - l_mut_16;
l_mut_24 = l_47;
}
l_mut_16 = l_mut_24;
l_arr_5[uint(0)] = l_mut_16;
const uint l_48 = uint(256) + thread_pos_in_threadgroup.x;
const uint l_49 = l_10 + l_48;
const float l_50 = buffer_0[l_49];
l_mut_24 = l_50;
const bool l_51 = l_2 == uint(0);
const bool l_52 = l_1 != uint(0);
const bool l_53 = l_51 && l_52;
if (l_53) {
const uint l_54 = l_12 + l_48;
const float l_55 = buffer_1[l_54];
l_mut_24 = l_mut_24 * l_55;
}
const float l_56 = { simd_shuffle_xor(l_mut_24, uint(1)) };
const float l_57 = l_mut_24 + l_56;
l_mut_16 = l_57;
const uint l_58 = thread_pos_in_threadgroup.x & uint(1);
const bool l_59 = l_58 != uint(0);
if (l_59) {
const float l_60 = l_56 - l_mut_24;
l_mut_16 = l_60;
}
l_mut_24 = l_mut_16;
const float l_61 = { simd_shuffle_xor(l_mut_24, uint(2)) };
const float l_62 = l_mut_24 + l_61;
l_mut_16 = l_62;
const uint l_63 = thread_pos_in_threadgroup.x & uint(2);
const bool l_64 = l_63 != uint(0);
if (l_64) {
const float l_65 = l_61 - l_mut_24;
l_mut_16 = l_65;
}
l_mut_24 = l_mut_16;
const float l_66 = { simd_shuffle_xor(l_mut_24, uint(4)) };
const float l_67 = l_mut_24 + l_66;
l_mut_16 = l_67;
const uint l_68 = thread_pos_in_threadgroup.x & uint(4);
const bool l_69 = l_68 != uint(0);
if (l_69) {
const float l_70 = l_66 - l_mut_24;
l_mut_16 = l_70;
}
l_mut_24 = l_mut_16;
const float l_71 = { simd_shuffle_xor(l_mut_24, uint(8)) };
const float l_72 = l_mut_24 + l_71;
l_mut_16 = l_72;
const uint l_73 = thread_pos_in_threadgroup.x & uint(8);
const bool l_74 = l_73 != uint(0);
if (l_74) {
const float l_75 = l_71 - l_mut_24;
l_mut_16 = l_75;
}
l_mut_24 = l_mut_16;
const float l_76 = { simd_shuffle_xor(l_mut_24, uint(16)) };
const float l_77 = l_mut_24 + l_76;
l_mut_16 = l_77;
const uint l_78 = thread_pos_in_threadgroup.x & uint(16);
const bool l_79 = l_78 != uint(0);
if (l_79) {
const float l_80 = l_76 - l_mut_24;
l_mut_16 = l_80;
}
l_mut_24 = l_mut_16;
l_arr_5[uint(1)] = l_mut_24;
const uint l_81 = uint(0) + thread_pos_in_threadgroup.x;
const float l_82 = l_arr_5[uint(0)];
shared_memory_4[l_81] = l_82;
const uint l_83 = uint(256) + thread_pos_in_threadgroup.x;
const float l_84 = l_arr_5[uint(1)];
shared_memory_4[l_83] = l_84;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_85 = thread_pos_in_threadgroup.x ^ uint(32);
const uint l_86 = thread_pos_in_threadgroup.x & uint(32);
const bool l_87 = l_86 != uint(0);
const float l_88 = l_arr_5[uint(0)];
const uint l_89 = uint(0) + l_85;
const float l_90 = shared_memory_4[l_89];
const float l_91 = l_88 + l_90;
l_mut_24 = l_91;
if (l_87) {
const float l_92 = l_90 - l_88;
l_mut_24 = l_92;
}
l_arr_6[uint(0)] = l_mut_24;
const float l_93 = l_arr_5[uint(1)];
const uint l_94 = uint(256) + l_85;
const float l_95 = shared_memory_4[l_94];
const float l_96 = l_93 + l_95;
l_mut_24 = l_96;
if (l_87) {
const float l_97 = l_95 - l_93;
l_mut_24 = l_97;
}
l_arr_6[uint(1)] = l_mut_24;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float l_98 = l_arr_6[uint(0)];
l_arr_5[uint(0)] = l_98;
const uint l_99 = uint(0) + thread_pos_in_threadgroup.x;
shared_memory_4[l_99] = l_98;
const float l_100 = l_arr_6[uint(1)];
l_arr_5[uint(1)] = l_100;
const uint l_101 = uint(256) + thread_pos_in_threadgroup.x;
shared_memory_4[l_101] = l_100;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_102 = thread_pos_in_threadgroup.x ^ uint(64);
const uint l_103 = thread_pos_in_threadgroup.x & uint(64);
const bool l_104 = l_103 != uint(0);
const float l_105 = l_arr_5[uint(0)];
const uint l_106 = uint(0) + l_102;
const float l_107 = shared_memory_4[l_106];
const float l_108 = l_105 + l_107;
l_mut_24 = l_108;
if (l_104) {
const float l_109 = l_107 - l_105;
l_mut_24 = l_109;
}
l_arr_6[uint(0)] = l_mut_24;
const float l_110 = l_arr_5[uint(1)];
const uint l_111 = uint(256) + l_102;
const float l_112 = shared_memory_4[l_111];
const float l_113 = l_110 + l_112;
l_mut_24 = l_113;
if (l_104) {
const float l_114 = l_112 - l_110;
l_mut_24 = l_114;
}
l_arr_6[uint(1)] = l_mut_24;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float l_115 = l_arr_6[uint(0)];
l_arr_5[uint(0)] = l_115;
const uint l_116 = uint(0) + thread_pos_in_threadgroup.x;
shared_memory_4[l_116] = l_115;
const float l_117 = l_arr_6[uint(1)];
l_arr_5[uint(1)] = l_117;
const uint l_118 = uint(256) + thread_pos_in_threadgroup.x;
shared_memory_4[l_118] = l_117;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_119 = thread_pos_in_threadgroup.x ^ uint(128);
const uint l_120 = thread_pos_in_threadgroup.x & uint(128);
const bool l_121 = l_120 != uint(0);
const float l_122 = l_arr_5[uint(0)];
const uint l_123 = uint(0) + l_119;
const float l_124 = shared_memory_4[l_123];
const float l_125 = l_122 + l_124;
l_mut_24 = l_125;
if (l_121) {
const float l_126 = l_124 - l_122;
l_mut_24 = l_126;
}
l_arr_6[uint(0)] = l_mut_24;
const float l_127 = l_arr_5[uint(1)];
const uint l_128 = uint(256) + l_119;
const float l_129 = shared_memory_4[l_128];
const float l_130 = l_127 + l_129;
l_mut_24 = l_130;
if (l_121) {
const float l_131 = l_129 - l_127;
l_mut_24 = l_131;
}
l_arr_6[uint(1)] = l_mut_24;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float l_132 = l_arr_6[uint(0)];
l_arr_5[uint(0)] = l_132;
const float l_133 = l_arr_6[uint(1)];
l_arr_5[uint(1)] = l_133;
const float l_134 = l_arr_5[uint(0)];
const float l_135 = l_arr_5[uint(1)];
const float l_136 = l_134 + l_135;
l_arr_5[uint(0)] = l_136;
const float l_137 = l_134 - l_135;
l_arr_5[uint(1)] = l_137;
const uint l_138 = uint(0) + thread_pos_in_threadgroup.x;
const float l_139 = l_arr_5[uint(0)];
const float l_140 = l_139 * float(0.04419417306780815);
l_mut_24 = l_140;
const bool l_141 = l_2 != uint(0);
const bool l_142 = l_1 != uint(0);
const bool l_143 = l_141 && l_142;
if (l_143) {
const uint l_144 = l_12 + l_138;
const float l_145 = buffer_1[l_144];
l_mut_24 = l_mut_24 * l_145;
}
const uint l_146 = l_10 + l_138;
buffer_2[l_146] = l_mut_24;
const uint l_147 = uint(256) + thread_pos_in_threadgroup.x;
const float l_148 = l_arr_5[uint(1)];
const float l_149 = l_148 * float(0.04419417306780815);
l_mut_24 = l_149;
const bool l_150 = l_2 != uint(0);
const bool l_151 = l_1 != uint(0);
const bool l_152 = l_150 && l_151;
if (l_152) {
const uint l_153 = l_12 + l_147;
const float l_154 = buffer_1[l_153];
l_mut_24 = l_mut_24 * l_154;
}
const uint l_155 = l_10 + l_147;
buffer_2[l_155] = l_mut_24;
l_mut_7 = l_mut_7 + threadgroups_per_grid.x;
}

}
