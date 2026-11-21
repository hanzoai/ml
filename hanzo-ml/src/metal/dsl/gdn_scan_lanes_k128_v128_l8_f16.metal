#include <metal_stdlib>
using namespace metal;

struct info_st {
    uint static_meta[16];
};

[[kernel]]
void gdn_scan_lanes_f_(
    const device half * buffer_0 [[buffer(0)]],
    const device half * buffer_1 [[buffer(1)]],
    const device half * buffer_2 [[buffer(2)]],
    const device half * buffer_3 [[buffer(3)]],
    const device half * buffer_4 [[buffer(4)]],
    device float * buffer_5 [[buffer(5)]],
    device half * buffer_6 [[buffer(6)]],
    const device uint * buffer_7 [[buffer(7)]],
    constant info_st& info [[buffer(8)]],
    uint3 thread_pos_in_grid [[thread_position_in_grid]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint simd_size [[threads_per_simdgroup]]
) {
uint thread_index_in_grid = (
                thread_pos_in_grid.z * threadgroups_per_grid.x * threads_per_threadgroup.x * threadgroups_per_grid.y * threads_per_threadgroup.y)
                + (thread_pos_in_grid.y * threadgroups_per_grid.x * threads_per_threadgroup.x)
                + thread_pos_in_grid.x;
uint threads_per_simdgroup_checked = min(simd_size, threads_per_threadgroup.x * threads_per_threadgroup.y * threads_per_threadgroup.z);
float l_arr_20[16];

uint l_mut_6;
float l_mut_40;
const uint l_0 = buffer_7[uint(0)];
const uint l_1 = buffer_7[uint(1)];
const uint l_2 = l_0 * uint(128);
const uint l_3 = thread_index_in_grid / uint(8);
const uint l_4 = thread_index_in_grid % uint(8);
const bool l_5 = l_3 < l_2;
l_mut_6 = l_3;
const bool l_7 = !l_5;
if (l_7) {
l_mut_6 = uint(0);
}
const uint l_8 = l_mut_6 / uint(128);
const uint l_9 = l_mut_6 % uint(128);
const uint l_10 = l_8 * l_1;
const uint l_11 = l_10 * uint(128);
const uint l_12 = l_4 * uint(16);
const uint l_13 = l_11 + l_12;
const uint l_14 = l_8 * l_1;
const uint l_15 = l_14 * uint(128);
const uint l_16 = l_15 + l_9;
const uint l_17 = l_8 * l_1;
const uint l_18 = l_8 * uint(128);
const uint l_19 = l_18 * uint(128);

for (uint l_mut_21 = uint(0); l_mut_21 < uint(16); ++l_mut_21) {
const uint l_22 = l_4 * uint(16);
const uint l_23 = l_22 + l_mut_21;
const uint l_24 = l_23 * uint(128);
const uint l_25 = l_19 + l_24;
const uint l_26 = l_25 + l_9;
const float l_27 = buffer_5[l_26];
l_arr_20[l_mut_21] = l_27;
}

for (uint l_mut_28 = uint(0); l_mut_28 < l_1; ++l_mut_28) {
const uint l_29 = l_17 + l_mut_28;
const half l_30 = buffer_3[l_29];
const float l_31 = float(l_30);
const float l_32 = exp(l_31);
const uint l_33 = l_17 + l_mut_28;
const half l_34 = buffer_4[l_33];
const float l_35 = float(l_34);
const uint l_36 = l_mut_28 * uint(128);
const uint l_37 = l_16 + l_36;
const half l_38 = buffer_2[l_37];
const float l_39 = float(l_38);
l_mut_40 = float(0.0);

for (uint l_mut_41 = uint(0); l_mut_41 < uint(16); ++l_mut_41) {
const float l_42 = l_arr_20[l_mut_41];
const float l_43 = l_42 * l_32;
l_arr_20[l_mut_41] = l_43;
const uint l_44 = l_mut_28 * uint(128);
const uint l_45 = l_13 + l_44;
const uint l_46 = l_45 + l_mut_41;
const half l_47 = buffer_1[l_46];
const float l_48 = float(l_47);
const float l_49 = l_43 * l_48;
l_mut_40 = l_mut_40 + l_49;
}
l_mut_6 = uint(1);
while (true) {
const bool l_50 = l_mut_6 < uint(8);
const bool l_51 = !l_50;
if (l_51) {
break;}
const float l_52 = { simd_shuffle_xor(l_mut_40, l_mut_6) };
l_mut_40 = l_mut_40 + l_52;
l_mut_6 = l_mut_6 * uint(2);
}
const float l_53 = l_39 - l_mut_40;
const float l_54 = l_53 * l_35;
l_mut_40 = float(0.0);

for (uint l_mut_55 = uint(0); l_mut_55 < uint(16); ++l_mut_55) {
const uint l_56 = l_mut_28 * uint(128);
const uint l_57 = l_13 + l_56;
const uint l_58 = l_57 + l_mut_55;
const half l_59 = buffer_1[l_58];
const float l_60 = float(l_59);
const float l_61 = l_arr_20[l_mut_55];
const float l_62 = l_60 * l_54;
const float l_63 = l_61 + l_62;
l_arr_20[l_mut_55] = l_63;
const uint l_64 = l_mut_28 * uint(128);
const uint l_65 = l_13 + l_64;
const uint l_66 = l_65 + l_mut_55;
const half l_67 = buffer_0[l_66];
const float l_68 = float(l_67);
const float l_69 = l_63 * l_68;
l_mut_40 = l_mut_40 + l_69;
}
l_mut_6 = uint(1);
while (true) {
const bool l_70 = l_mut_6 < uint(8);
const bool l_71 = !l_70;
if (l_71) {
break;}
const float l_72 = { simd_shuffle_xor(l_mut_40, l_mut_6) };
l_mut_40 = l_mut_40 + l_72;
l_mut_6 = l_mut_6 * uint(2);
}
const bool l_73 = l_4 == uint(0);
const bool l_74 = l_5 && l_73;
if (l_74) {
const uint l_75 = l_mut_28 * uint(128);
const uint l_76 = l_16 + l_75;
const half l_77 = half(l_mut_40);
buffer_6[l_76] = l_77;
}
}
if (l_5) {

for (uint l_mut_78 = uint(0); l_mut_78 < uint(16); ++l_mut_78) {
const uint l_79 = l_4 * uint(16);
const uint l_80 = l_79 + l_mut_78;
const uint l_81 = l_80 * uint(128);
const uint l_82 = l_19 + l_81;
const uint l_83 = l_82 + l_9;
const float l_84 = l_arr_20[l_mut_78];
buffer_5[l_83] = l_84;
}
}

}
