#include <metal_stdlib>
using namespace metal;

struct info_st {
    uint static_meta[10];
};

[[kernel]]
void gdn_conv1d_step_f_(
    const device bfloat * buffer_0 [[buffer(0)]],
    const device bfloat * buffer_1 [[buffer(1)]],
    device bfloat * buffer_2 [[buffer(2)]],
    device bfloat * buffer_3 [[buffer(3)]],
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

for (uint l_mut_8 = uint(0); l_mut_8 < uint(3); ++l_mut_8) {
const uint l_9 = l_5 + l_mut_8;
const uint l_10 = l_9 + uint(1);
const bfloat l_11 = buffer_2[l_10];
const uint l_12 = l_5 + l_mut_8;
buffer_2[l_12] = l_11;
const float l_13 = float(l_11);
const uint l_14 = l_6 + l_mut_8;
const bfloat l_15 = buffer_1[l_14];
const float l_16 = float(l_15);
const float l_17 = l_13 * l_16;
l_mut_7 = l_mut_7 + l_17;
}
const bfloat l_18 = buffer_0[thread_index_in_grid];
const uint l_19 = l_5 + uint(4);
const uint l_20 = l_19 - uint(1);
buffer_2[l_20] = l_18;
const float l_21 = float(l_18);
const uint l_22 = l_6 + uint(4);
const uint l_23 = l_22 - uint(1);
const bfloat l_24 = buffer_1[l_23];
const float l_25 = float(l_24);
const float l_26 = l_21 * l_25;
l_mut_7 = l_mut_7 + l_26;
const float l_27 = -l_mut_7;
const float l_28 = exp(l_27);
const float l_29 = float(1.0) + l_28;
const float l_30 = l_mut_7 / l_29;
const bfloat l_31 = bfloat(l_30);
buffer_3[thread_index_in_grid] = l_31;
}

}
