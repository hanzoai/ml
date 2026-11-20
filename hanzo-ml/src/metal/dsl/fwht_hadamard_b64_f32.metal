#include <metal_stdlib>
using namespace metal;

struct info_st {
    uint static_meta[8];
};

[[kernel]]
void fwht_hadamard_f_f32(
    const device float * buffer_0 [[buffer(0)]],
    const device float * buffer_1 [[buffer(1)]],
    device float * buffer_2 [[buffer(2)]],
    const device uint * buffer_3 [[buffer(3)]],
    constant info_st& info [[buffer(4)]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 thread_pos_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_pos_in_grid [[threadgroup_position_in_grid]]
) {
uint threadgroup_index_in_grid = (threadgroup_pos_in_grid.z * threadgroups_per_grid.y * threadgroups_per_grid.x) + (threadgroup_pos_in_grid.y * threadgroups_per_grid.x) + threadgroup_pos_in_grid.x;
threadgroup uchar dynamic_shared_mem[256];
// Shared array size: 64, 256 bytes
threadgroup float* shared_memory_9 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[0]);
uint l_mut_6;
float l_mut_14;
float l_mut_18;
float l_mut_22;
float l_mut_26;
const uint l_0 = buffer_3[uint(0)];
const bool l_1 = threadgroup_index_in_grid < l_0;
if (l_1) {
const uint l_2 = buffer_3[uint(1)];
const uint l_3 = buffer_3[uint(2)];
const uint l_4 = buffer_3[uint(3)];
const uint l_5 = threadgroup_index_in_grid * uint(64);
l_mut_6 = uint(0);
const bool l_7 = l_3 != uint(0);
if (l_7) {
const uint l_8 = l_5 % l_2;
l_mut_6 = l_8;
}
const uint l_10 = thread_index_in_threadgroup + uint(0);
const uint l_11 = l_10 * uint(4);
const uint l_12 = l_5 + l_11;
const float l_13 = buffer_0[l_12];
l_mut_14 = l_13;
const uint l_15 = l_5 + l_11;
const uint l_16 = l_15 + uint(1);
const float l_17 = buffer_0[l_16];
l_mut_18 = l_17;
const uint l_19 = l_5 + l_11;
const uint l_20 = l_19 + uint(2);
const float l_21 = buffer_0[l_20];
l_mut_22 = l_21;
const uint l_23 = l_5 + l_11;
const uint l_24 = l_23 + uint(3);
const float l_25 = buffer_0[l_24];
l_mut_26 = l_25;
const bool l_27 = l_3 != uint(0);
const bool l_28 = l_4 == uint(0);
const bool l_29 = l_27 && l_28;
if (l_29) {
const uint l_30 = l_mut_6 + l_11;
const float l_31 = buffer_1[l_30];
l_mut_14 = l_mut_14 * l_31;
const uint l_32 = l_mut_6 + l_11;
const uint l_33 = l_32 + uint(1);
const float l_34 = buffer_1[l_33];
l_mut_18 = l_mut_18 * l_34;
const uint l_35 = l_mut_6 + l_11;
const uint l_36 = l_35 + uint(2);
const float l_37 = buffer_1[l_36];
l_mut_22 = l_mut_22 * l_37;
const uint l_38 = l_mut_6 + l_11;
const uint l_39 = l_38 + uint(3);
const float l_40 = buffer_1[l_39];
l_mut_26 = l_mut_26 * l_40;
}
const float l_41 = l_mut_14 + l_mut_18;
const float l_42 = l_mut_14 - l_mut_18;
const float l_43 = l_mut_22 + l_mut_26;
const float l_44 = l_mut_22 - l_mut_26;
const float l_45 = l_41 + l_43;
shared_memory_9[l_11] = l_45;
const uint l_46 = l_11 + uint(1);
const float l_47 = l_42 + l_44;
shared_memory_9[l_46] = l_47;
const uint l_48 = l_11 + uint(2);
const float l_49 = l_41 - l_43;
shared_memory_9[l_48] = l_49;
const uint l_50 = l_11 + uint(3);
const float l_51 = l_42 - l_44;
shared_memory_9[l_50] = l_51;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_52 = thread_index_in_threadgroup + uint(0);
const uint l_53 = l_52 >> uint(2);
const uint l_54 = l_52 & uint(3);
const uint l_55 = l_53 * uint(8);
const uint l_56 = l_55 + l_54;
const uint l_57 = l_56 + uint(4);
const float l_58 = shared_memory_9[l_56];
const float l_59 = shared_memory_9[l_57];
const float l_60 = l_58 + l_59;
shared_memory_9[l_56] = l_60;
const float l_61 = l_58 - l_59;
shared_memory_9[l_57] = l_61;
const uint l_62 = thread_index_in_threadgroup + uint(16);
const uint l_63 = l_62 >> uint(2);
const uint l_64 = l_62 & uint(3);
const uint l_65 = l_63 * uint(8);
const uint l_66 = l_65 + l_64;
const uint l_67 = l_66 + uint(4);
const float l_68 = shared_memory_9[l_66];
const float l_69 = shared_memory_9[l_67];
const float l_70 = l_68 + l_69;
shared_memory_9[l_66] = l_70;
const float l_71 = l_68 - l_69;
shared_memory_9[l_67] = l_71;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_72 = thread_index_in_threadgroup + uint(0);
const uint l_73 = l_72 >> uint(3);
const uint l_74 = l_72 & uint(7);
const uint l_75 = l_73 * uint(16);
const uint l_76 = l_75 + l_74;
const uint l_77 = l_76 + uint(8);
const float l_78 = shared_memory_9[l_76];
const float l_79 = shared_memory_9[l_77];
const float l_80 = l_78 + l_79;
shared_memory_9[l_76] = l_80;
const float l_81 = l_78 - l_79;
shared_memory_9[l_77] = l_81;
const uint l_82 = thread_index_in_threadgroup + uint(16);
const uint l_83 = l_82 >> uint(3);
const uint l_84 = l_82 & uint(7);
const uint l_85 = l_83 * uint(16);
const uint l_86 = l_85 + l_84;
const uint l_87 = l_86 + uint(8);
const float l_88 = shared_memory_9[l_86];
const float l_89 = shared_memory_9[l_87];
const float l_90 = l_88 + l_89;
shared_memory_9[l_86] = l_90;
const float l_91 = l_88 - l_89;
shared_memory_9[l_87] = l_91;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_92 = thread_index_in_threadgroup + uint(0);
const uint l_93 = l_92 >> uint(4);
const uint l_94 = l_92 & uint(15);
const uint l_95 = l_93 * uint(32);
const uint l_96 = l_95 + l_94;
const uint l_97 = l_96 + uint(16);
const float l_98 = shared_memory_9[l_96];
const float l_99 = shared_memory_9[l_97];
const float l_100 = l_98 + l_99;
shared_memory_9[l_96] = l_100;
const float l_101 = l_98 - l_99;
shared_memory_9[l_97] = l_101;
const uint l_102 = thread_index_in_threadgroup + uint(16);
const uint l_103 = l_102 >> uint(4);
const uint l_104 = l_102 & uint(15);
const uint l_105 = l_103 * uint(32);
const uint l_106 = l_105 + l_104;
const uint l_107 = l_106 + uint(16);
const float l_108 = shared_memory_9[l_106];
const float l_109 = shared_memory_9[l_107];
const float l_110 = l_108 + l_109;
shared_memory_9[l_106] = l_110;
const float l_111 = l_108 - l_109;
shared_memory_9[l_107] = l_111;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_112 = thread_index_in_threadgroup + uint(0);
const uint l_113 = l_112 >> uint(5);
const uint l_114 = l_112 & uint(31);
const uint l_115 = l_113 * uint(64);
const uint l_116 = l_115 + l_114;
const uint l_117 = l_116 + uint(32);
const float l_118 = shared_memory_9[l_116];
const float l_119 = shared_memory_9[l_117];
const float l_120 = l_118 + l_119;
shared_memory_9[l_116] = l_120;
const float l_121 = l_118 - l_119;
shared_memory_9[l_117] = l_121;
const uint l_122 = thread_index_in_threadgroup + uint(16);
const uint l_123 = l_122 >> uint(5);
const uint l_124 = l_122 & uint(31);
const uint l_125 = l_123 * uint(64);
const uint l_126 = l_125 + l_124;
const uint l_127 = l_126 + uint(32);
const float l_128 = shared_memory_9[l_126];
const float l_129 = shared_memory_9[l_127];
const float l_130 = l_128 + l_129;
shared_memory_9[l_126] = l_130;
const float l_131 = l_128 - l_129;
shared_memory_9[l_127] = l_131;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_132 = thread_index_in_threadgroup + uint(0);
const uint l_133 = l_132 * uint(4);
const float l_134 = shared_memory_9[l_133];
const float l_135 = l_134 * float(0.125);
l_mut_26 = l_135;
const uint l_136 = l_133 + uint(1);
const float l_137 = shared_memory_9[l_136];
const float l_138 = l_137 * float(0.125);
l_mut_22 = l_138;
const uint l_139 = l_133 + uint(2);
const float l_140 = shared_memory_9[l_139];
const float l_141 = l_140 * float(0.125);
l_mut_18 = l_141;
const uint l_142 = l_133 + uint(3);
const float l_143 = shared_memory_9[l_142];
const float l_144 = l_143 * float(0.125);
l_mut_14 = l_144;
const bool l_145 = l_3 != uint(0);
const bool l_146 = l_4 != uint(0);
const bool l_147 = l_145 && l_146;
if (l_147) {
const uint l_148 = l_mut_6 + l_133;
const float l_149 = buffer_1[l_148];
l_mut_26 = l_mut_26 * l_149;
const uint l_150 = l_mut_6 + l_133;
const uint l_151 = l_150 + uint(1);
const float l_152 = buffer_1[l_151];
l_mut_22 = l_mut_22 * l_152;
const uint l_153 = l_mut_6 + l_133;
const uint l_154 = l_153 + uint(2);
const float l_155 = buffer_1[l_154];
l_mut_18 = l_mut_18 * l_155;
const uint l_156 = l_mut_6 + l_133;
const uint l_157 = l_156 + uint(3);
const float l_158 = buffer_1[l_157];
l_mut_14 = l_mut_14 * l_158;
}
const uint l_159 = l_5 + l_133;
buffer_2[l_159] = l_mut_26;
const uint l_160 = l_5 + l_133;
const uint l_161 = l_160 + uint(1);
buffer_2[l_161] = l_mut_22;
const uint l_162 = l_5 + l_133;
const uint l_163 = l_162 + uint(2);
buffer_2[l_163] = l_mut_18;
const uint l_164 = l_5 + l_133;
const uint l_165 = l_164 + uint(3);
buffer_2[l_165] = l_mut_14;
}

}
