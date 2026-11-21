#include <metal_stdlib>
using namespace metal;

struct info_st {
    uint static_meta[16];
};

[[kernel]]
void gdn_scan_lanes_f_f32(
    const device float * buffer_0 [[buffer(0)]],
    const device float * buffer_1 [[buffer(1)]],
    const device float * buffer_2 [[buffer(2)]],
    const device float * buffer_3 [[buffer(3)]],
    const device float * buffer_4 [[buffer(4)]],
    device float * buffer_5 [[buffer(5)]],
    device float * buffer_6 [[buffer(6)]],
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
float l_arr_20[8];

uint l_mut_6;
float l_mut_37;
const uint l_0 = buffer_7[uint(0)];
const uint l_1 = buffer_7[uint(1)];
const uint l_2 = l_0 * uint(128);
const uint l_3 = thread_index_in_grid / uint(16);
const uint l_4 = thread_index_in_grid % uint(16);
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
const uint l_12 = l_4 * uint(8);
const uint l_13 = l_11 + l_12;
const uint l_14 = l_8 * l_1;
const uint l_15 = l_14 * uint(128);
const uint l_16 = l_15 + l_9;
const uint l_17 = l_8 * l_1;
const uint l_18 = l_8 * uint(128);
const uint l_19 = l_18 * uint(128);

for (uint l_mut_21 = uint(0); l_mut_21 < uint(8); ++l_mut_21) {
const uint l_22 = l_4 * uint(8);
const uint l_23 = l_22 + l_mut_21;
const uint l_24 = l_23 * uint(128);
const uint l_25 = l_19 + l_24;
const uint l_26 = l_25 + l_9;
const float l_27 = buffer_5[l_26];
l_arr_20[l_mut_21] = l_27;
}

for (uint l_mut_28 = uint(0); l_mut_28 < l_1; ++l_mut_28) {
const uint l_29 = l_17 + l_mut_28;
const float l_30 = buffer_3[l_29];
const float l_31 = exp(l_30);
const uint l_32 = l_17 + l_mut_28;
const float l_33 = buffer_4[l_32];
const uint l_34 = l_mut_28 * uint(128);
const uint l_35 = l_16 + l_34;
const float l_36 = buffer_2[l_35];
l_mut_37 = float(0.0);

for (uint l_mut_38 = uint(0); l_mut_38 < uint(8); ++l_mut_38) {
const float l_39 = l_arr_20[l_mut_38];
const float l_40 = l_39 * l_31;
l_arr_20[l_mut_38] = l_40;
const uint l_41 = l_mut_28 * uint(128);
const uint l_42 = l_13 + l_41;
const uint l_43 = l_42 + l_mut_38;
const float l_44 = buffer_1[l_43];
const float l_45 = l_40 * l_44;
l_mut_37 = l_mut_37 + l_45;
}
l_mut_6 = uint(1);
while (true) {
const bool l_46 = l_mut_6 < uint(16);
const bool l_47 = !l_46;
if (l_47) {
break;}
const float l_48 = { simd_shuffle_xor(l_mut_37, l_mut_6) };
l_mut_37 = l_mut_37 + l_48;
l_mut_6 = l_mut_6 * uint(2);
}
const float l_49 = l_36 - l_mut_37;
const float l_50 = l_49 * l_33;
l_mut_37 = float(0.0);

for (uint l_mut_51 = uint(0); l_mut_51 < uint(8); ++l_mut_51) {
const uint l_52 = l_mut_28 * uint(128);
const uint l_53 = l_13 + l_52;
const uint l_54 = l_53 + l_mut_51;
const float l_55 = buffer_1[l_54];
const float l_56 = l_arr_20[l_mut_51];
const float l_57 = l_55 * l_50;
const float l_58 = l_56 + l_57;
l_arr_20[l_mut_51] = l_58;
const uint l_59 = l_mut_28 * uint(128);
const uint l_60 = l_13 + l_59;
const uint l_61 = l_60 + l_mut_51;
const float l_62 = buffer_0[l_61];
const float l_63 = l_58 * l_62;
l_mut_37 = l_mut_37 + l_63;
}
l_mut_6 = uint(1);
while (true) {
const bool l_64 = l_mut_6 < uint(16);
const bool l_65 = !l_64;
if (l_65) {
break;}
const float l_66 = { simd_shuffle_xor(l_mut_37, l_mut_6) };
l_mut_37 = l_mut_37 + l_66;
l_mut_6 = l_mut_6 * uint(2);
}
const bool l_67 = l_4 == uint(0);
const bool l_68 = l_5 && l_67;
if (l_68) {
const uint l_69 = l_mut_28 * uint(128);
const uint l_70 = l_16 + l_69;
buffer_6[l_70] = l_mut_37;
}
}
if (l_5) {

for (uint l_mut_71 = uint(0); l_mut_71 < uint(8); ++l_mut_71) {
const uint l_72 = l_4 * uint(8);
const uint l_73 = l_72 + l_mut_71;
const uint l_74 = l_73 * uint(128);
const uint l_75 = l_19 + l_74;
const uint l_76 = l_75 + l_9;
const float l_77 = l_arr_20[l_mut_71];
buffer_5[l_76] = l_77;
}
}

}
