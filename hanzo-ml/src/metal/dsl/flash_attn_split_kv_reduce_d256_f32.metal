#include <metal_stdlib>
using namespace metal;

struct info_st {
    uint static_meta[6];
};

[[kernel]]
void flash_attn_split_kv_reduce_f_f32(
    const device float * buffer_0 [[buffer(0)]],
    device float * buffer_1 [[buffer(1)]],
    const device uint * buffer_2 [[buffer(2)]],
    constant info_st& info [[buffer(3)]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 thread_pos_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_pos_in_grid [[threadgroup_position_in_grid]]
) {
uint threadgroup_index_in_grid = (threadgroup_pos_in_grid.z * threadgroups_per_grid.y * threadgroups_per_grid.x) + (threadgroup_pos_in_grid.y * threadgroups_per_grid.x) + threadgroup_pos_in_grid.x;
float l_arr_7[8];

float l_mut_5;
float l_mut_6;
float l_mut_20;
const uint l_0 = buffer_2[uint(2)];
const uint l_1 = threadgroup_index_in_grid + l_0;
const uint l_2 = buffer_2[uint(0)];
const uint l_3 = buffer_2[uint(1)];
const bool l_4 = l_1 < l_2;
if (l_4) {
l_mut_5 = float(-3.3999999521443642e38);
l_mut_6 = float(0.0);

for (uint l_mut_8 = uint(0); l_mut_8 < uint(8); ++l_mut_8) {
l_arr_7[l_mut_8] = float(0.0);
}
const uint l_9 = l_1 * l_3;
const uint l_10 = l_9 * uint(258);

for (uint l_mut_11 = uint(0); l_mut_11 < l_3; ++l_mut_11) {
const uint l_12 = l_mut_11 * uint(258);
const uint l_13 = l_10 + l_12;
const uint l_14 = l_13 + uint(256);
const float l_15 = buffer_0[l_14];
const uint l_16 = l_13 + uint(256);
const uint l_17 = l_16 + uint(1);
const float l_18 = buffer_0[l_17];
const bool l_19 = l_18 > float(0.0);
if (l_19) {
l_mut_20 = l_mut_5;
const bool l_21 = l_15 > l_mut_20;
if (l_21) {
l_mut_20 = l_15;
}
const float l_22 = l_mut_5 - l_mut_20;
const float l_23 = exp(l_22);
const float l_24 = l_15 - l_mut_20;
const float l_25 = exp(l_24);
const float l_26 = l_mut_6 * l_23;
const float l_27 = l_18 * l_25;
const float l_28 = l_26 + l_27;
l_mut_6 = l_28;

for (uint l_mut_29 = uint(0); l_mut_29 < uint(8); ++l_mut_29) {
const uint l_30 = l_mut_29 * uint(32);
const uint l_31 = l_30 + thread_index_in_threadgroup;
const float l_32 = l_arr_7[l_mut_29];
const float l_33 = l_32 * l_23;
const uint l_34 = l_13 + l_31;
const float l_35 = buffer_0[l_34];
const float l_36 = l_35 * l_25;
const float l_37 = l_33 + l_36;
l_arr_7[l_mut_29] = l_37;
}
l_mut_5 = l_mut_20;
}
}
l_mut_20 = float(0.0);
const bool l_38 = l_mut_6 > float(0.0);
if (l_38) {
const float l_39 = float(1.0) / l_mut_6;
l_mut_20 = l_39;
}
const uint l_40 = l_1 * uint(256);

for (uint l_mut_41 = uint(0); l_mut_41 < uint(8); ++l_mut_41) {
const uint l_42 = l_mut_41 * uint(32);
const uint l_43 = l_42 + thread_index_in_threadgroup;
const uint l_44 = l_40 + l_43;
const float l_45 = l_arr_7[l_mut_41];
const float l_46 = l_45 * l_mut_20;
buffer_1[l_44] = l_46;
}
}

}
