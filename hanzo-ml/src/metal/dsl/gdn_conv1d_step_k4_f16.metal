#include <metal_stdlib>
using namespace metal;

struct info_st {
    uint static_meta[10];
};

[[kernel]]
void gdn_conv1d_step_f_(
    const device half * buffer_0 [[buffer(0)]],
    const device half * buffer_1 [[buffer(1)]],
    device half * buffer_2 [[buffer(2)]],
    device half * buffer_3 [[buffer(3)]],
    const device uint * buffer_4 [[buffer(4)]],
    constant info_st& info [[buffer(5)]],
    uint3 thread_pos_in_grid [[thread_position_in_grid]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]]
) {
uint thread_index_in_grid = (
                thread_pos_in_grid.z * threadgroups_per_grid.x * threads_per_threadgroup.x * threadgroups_per_grid.y * threads_per_threadgroup.y)
                + (thread_pos_in_grid.y * threadgroups_per_grid.x * threads_per_threadgroup.x)
                + thread_pos_in_grid.x;
float l_mut_7;
const uint l_0 = buffer_4[uint(0)];
const uint l_1 = buffer_4[uint(1)];
const uint l_2 = l_0 * l_1;
const bool l_3 = thread_index_in_grid < l_2;
if (l_3) {
const uint l_4 = thread_index_in_grid % l_1;
const uint l_5 = thread_index_in_grid * uint(4);
const uint l_6 = l_4 * uint(4);
l_mut_7 = float(0.0);
const uint l_8 = l_5 + uint(0);
const uint l_9 = l_8 + uint(1);
const half l_10 = buffer_2[l_9];
const uint l_11 = l_5 + uint(0);
buffer_2[l_11] = l_10;
const float l_12 = float(l_10);
const uint l_13 = l_6 + uint(0);
const half l_14 = buffer_1[l_13];
const float l_15 = float(l_14);
const float l_16 = l_12 * l_15;
l_mut_7 = l_mut_7 + l_16;
const uint l_17 = l_5 + uint(1);
const uint l_18 = l_17 + uint(1);
const half l_19 = buffer_2[l_18];
const uint l_20 = l_5 + uint(1);
buffer_2[l_20] = l_19;
const float l_21 = float(l_19);
const uint l_22 = l_6 + uint(1);
const half l_23 = buffer_1[l_22];
const float l_24 = float(l_23);
const float l_25 = l_21 * l_24;
l_mut_7 = l_mut_7 + l_25;
const uint l_26 = l_5 + uint(2);
const uint l_27 = l_26 + uint(1);
const half l_28 = buffer_2[l_27];
const uint l_29 = l_5 + uint(2);
buffer_2[l_29] = l_28;
const float l_30 = float(l_28);
const uint l_31 = l_6 + uint(2);
const half l_32 = buffer_1[l_31];
const float l_33 = float(l_32);
const float l_34 = l_30 * l_33;
l_mut_7 = l_mut_7 + l_34;
const half l_35 = buffer_0[thread_index_in_grid];
const uint l_36 = l_5 + uint(3);
buffer_2[l_36] = l_35;
const float l_37 = float(l_35);
const uint l_38 = l_6 + uint(3);
const half l_39 = buffer_1[l_38];
const float l_40 = float(l_39);
const float l_41 = l_37 * l_40;
l_mut_7 = l_mut_7 + l_41;
const float l_42 = -l_mut_7;
const float l_43 = exp(l_42);
const float l_44 = float(1.0) + l_43;
const float l_45 = l_mut_7 / l_44;
const half l_46 = half(l_45);
buffer_3[thread_index_in_grid] = l_46;
}

}
