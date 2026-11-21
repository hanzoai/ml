#include <metal_stdlib>
using namespace metal;

struct info_st {
    uint static_meta[12];
};

[[kernel]]
void gdn_conv1d_stateful_f_(
    const device half * buffer_0 [[buffer(0)]],
    const device half * buffer_1 [[buffer(1)]],
    const device half * buffer_2 [[buffer(2)]],
    device half * buffer_3 [[buffer(3)]],
    device half * buffer_4 [[buffer(4)]],
    const device uint * buffer_5 [[buffer(5)]],
    constant info_st& info [[buffer(6)]],
    uint3 thread_pos_in_grid [[thread_position_in_grid]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]]
) {
uint thread_index_in_grid = (
                thread_pos_in_grid.z * threadgroups_per_grid.x * threads_per_threadgroup.x * threadgroups_per_grid.y * threads_per_threadgroup.y)
                + (thread_pos_in_grid.y * threadgroups_per_grid.x * threads_per_threadgroup.x)
                + thread_pos_in_grid.x;
float l_mut_13;
float l_mut_14;
half l_mut_97;
half l_mut_102;
const uint l_0 = buffer_5[uint(0)];
const uint l_1 = buffer_5[uint(1)];
const uint l_2 = buffer_5[uint(2)];
const uint l_3 = buffer_5[uint(3)];
const uint l_4 = l_0 * l_1;
const uint l_5 = l_4 * l_2;
const bool l_6 = thread_index_in_grid < l_5;
if (l_6) {
const uint l_7 = thread_index_in_grid % l_2;
const uint l_8 = thread_index_in_grid / l_2;
const uint l_9 = l_8 % l_1;
const uint l_10 = l_8 * l_2;
const uint l_11 = l_8 * uint(4);
const uint l_12 = l_9 * uint(4);
l_mut_13 = float(0.0);
l_mut_14 = float(0.0);
const uint l_15 = l_7 + uint(0);
const bool l_16 = l_15 >= uint(3);
if (l_16) {
const uint l_17 = l_10 + l_7;
const uint l_18 = l_17 + uint(0);
const uint l_19 = l_18 - uint(3);
const half l_20 = buffer_0[l_19];
const float l_21 = float(l_20);
l_mut_14 = l_21;
} else {
const bool l_22 = l_3 != uint(0);
if (l_22) {
const uint l_23 = l_11 + uint(1);
const uint l_24 = l_23 + l_7;
const uint l_25 = l_24 + uint(0);
const half l_26 = buffer_2[l_25];
const float l_27 = float(l_26);
l_mut_14 = l_27;
}
}
const uint l_28 = l_12 + uint(0);
const half l_29 = buffer_1[l_28];
const float l_30 = float(l_29);
const float l_31 = l_mut_14 * l_30;
l_mut_13 = l_mut_13 + l_31;
l_mut_14 = float(0.0);
const uint l_32 = l_7 + uint(1);
const bool l_33 = l_32 >= uint(3);
if (l_33) {
const uint l_34 = l_10 + l_7;
const uint l_35 = l_34 + uint(1);
const uint l_36 = l_35 - uint(3);
const half l_37 = buffer_0[l_36];
const float l_38 = float(l_37);
l_mut_14 = l_38;
} else {
const bool l_39 = l_3 != uint(0);
if (l_39) {
const uint l_40 = l_11 + uint(1);
const uint l_41 = l_40 + l_7;
const uint l_42 = l_41 + uint(1);
const half l_43 = buffer_2[l_42];
const float l_44 = float(l_43);
l_mut_14 = l_44;
}
}
const uint l_45 = l_12 + uint(1);
const half l_46 = buffer_1[l_45];
const float l_47 = float(l_46);
const float l_48 = l_mut_14 * l_47;
l_mut_13 = l_mut_13 + l_48;
l_mut_14 = float(0.0);
const uint l_49 = l_7 + uint(2);
const bool l_50 = l_49 >= uint(3);
if (l_50) {
const uint l_51 = l_10 + l_7;
const uint l_52 = l_51 + uint(2);
const uint l_53 = l_52 - uint(3);
const half l_54 = buffer_0[l_53];
const float l_55 = float(l_54);
l_mut_14 = l_55;
} else {
const bool l_56 = l_3 != uint(0);
if (l_56) {
const uint l_57 = l_11 + uint(1);
const uint l_58 = l_57 + l_7;
const uint l_59 = l_58 + uint(2);
const half l_60 = buffer_2[l_59];
const float l_61 = float(l_60);
l_mut_14 = l_61;
}
}
const uint l_62 = l_12 + uint(2);
const half l_63 = buffer_1[l_62];
const float l_64 = float(l_63);
const float l_65 = l_mut_14 * l_64;
l_mut_13 = l_mut_13 + l_65;
l_mut_14 = float(0.0);
const uint l_66 = l_7 + uint(3);
const bool l_67 = l_66 >= uint(3);
if (l_67) {
const uint l_68 = l_10 + l_7;
const uint l_69 = l_68 + uint(3);
const uint l_70 = l_69 - uint(3);
const half l_71 = buffer_0[l_70];
const float l_72 = float(l_71);
l_mut_14 = l_72;
} else {
const bool l_73 = l_3 != uint(0);
if (l_73) {
const uint l_74 = l_11 + uint(1);
const uint l_75 = l_74 + l_7;
const uint l_76 = l_75 + uint(3);
const half l_77 = buffer_2[l_76];
const float l_78 = float(l_77);
l_mut_14 = l_78;
}
}
const uint l_79 = l_12 + uint(3);
const half l_80 = buffer_1[l_79];
const float l_81 = float(l_80);
const float l_82 = l_mut_14 * l_81;
l_mut_13 = l_mut_13 + l_82;
const uint l_83 = l_10 + l_7;
const float l_84 = -l_mut_13;
const float l_85 = exp(l_84);
const float l_86 = float(1.0) + l_85;
const float l_87 = l_mut_13 / l_86;
const half l_88 = half(l_87);
buffer_4[l_83] = l_88;
const uint l_89 = l_7 + uint(1);
const bool l_90 = l_89 == l_2;
if (l_90) {
const uint l_91 = l_2 + uint(0);
const bool l_92 = l_91 >= uint(4);
if (l_92) {
const uint l_93 = l_10 + l_2;
const uint l_94 = l_93 + uint(0);
const uint l_95 = l_94 - uint(4);
const half l_96 = buffer_0[l_95];
l_mut_97 = l_96;
} else {
const bool l_98 = l_3 != uint(0);
if (l_98) {
const uint l_99 = l_11 + l_2;
const uint l_100 = l_99 + uint(0);
const half l_101 = buffer_2[l_100];
l_mut_102 = l_101;
} else {
l_mut_102 = half(0.0);
}
l_mut_97 = l_mut_102;
}
const uint l_103 = l_11 + uint(0);
buffer_3[l_103] = l_mut_97;
const uint l_104 = l_2 + uint(1);
const bool l_105 = l_104 >= uint(4);
if (l_105) {
const uint l_106 = l_10 + l_2;
const uint l_107 = l_106 + uint(1);
const uint l_108 = l_107 - uint(4);
const half l_109 = buffer_0[l_108];
l_mut_102 = l_109;
} else {
const bool l_110 = l_3 != uint(0);
if (l_110) {
const uint l_111 = l_11 + l_2;
const uint l_112 = l_111 + uint(1);
const half l_113 = buffer_2[l_112];
l_mut_97 = l_113;
} else {
l_mut_97 = half(0.0);
}
l_mut_102 = l_mut_97;
}
const uint l_114 = l_11 + uint(1);
buffer_3[l_114] = l_mut_102;
const uint l_115 = l_2 + uint(2);
const bool l_116 = l_115 >= uint(4);
if (l_116) {
const uint l_117 = l_10 + l_2;
const uint l_118 = l_117 + uint(2);
const uint l_119 = l_118 - uint(4);
const half l_120 = buffer_0[l_119];
l_mut_102 = l_120;
} else {
const bool l_121 = l_3 != uint(0);
if (l_121) {
const uint l_122 = l_11 + l_2;
const uint l_123 = l_122 + uint(2);
const half l_124 = buffer_2[l_123];
l_mut_97 = l_124;
} else {
l_mut_97 = half(0.0);
}
l_mut_102 = l_mut_97;
}
const uint l_125 = l_11 + uint(2);
buffer_3[l_125] = l_mut_102;
const uint l_126 = l_2 + uint(3);
const bool l_127 = l_126 >= uint(4);
if (l_127) {
const uint l_128 = l_10 + l_2;
const uint l_129 = l_128 + uint(3);
const uint l_130 = l_129 - uint(4);
const half l_131 = buffer_0[l_130];
l_mut_102 = l_131;
} else {
const bool l_132 = l_3 != uint(0);
if (l_132) {
const uint l_133 = l_11 + l_2;
const uint l_134 = l_133 + uint(3);
const half l_135 = buffer_2[l_134];
l_mut_97 = l_135;
} else {
l_mut_97 = half(0.0);
}
l_mut_102 = l_mut_97;
}
const uint l_136 = l_11 + uint(3);
buffer_3[l_136] = l_mut_102;
}
}

}
