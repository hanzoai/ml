#include <metal_stdlib>
using namespace metal;

struct info_st {
    uint static_meta[8];
};

[[kernel]]
void gdn_conv1d_f_f32(
    const device float * buffer_0 [[buffer(0)]],
    const device float * buffer_1 [[buffer(1)]],
    device float * buffer_2 [[buffer(2)]],
    const device uint * buffer_3 [[buffer(3)]],
    constant info_st& info [[buffer(4)]],
    uint3 thread_pos_in_grid [[thread_position_in_grid]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]]
) {
uint thread_index_in_grid = (
                thread_pos_in_grid.z * threadgroups_per_grid.x * threads_per_threadgroup.x * threadgroups_per_grid.y * threads_per_threadgroup.y)
                + (thread_pos_in_grid.y * threadgroups_per_grid.x * threads_per_threadgroup.x)
                + thread_pos_in_grid.x;
float l_mut_9;
const uint l_0 = info.static_meta[uint(6)];
const bool l_1 = thread_index_in_grid < l_0;
if (l_1) {
const uint l_2 = buffer_3[uint(0)];
const uint l_3 = buffer_3[uint(1)];
const uint l_4 = thread_index_in_grid % l_3;
const uint l_5 = thread_index_in_grid / l_3;
const uint l_6 = l_5 % l_2;
const uint l_7 = l_5 * l_3;
const uint l_8 = l_6 * uint(4);
l_mut_9 = float(0.0);

for (uint l_mut_10 = uint(0); l_mut_10 < uint(4); ++l_mut_10) {
const uint l_11 = l_4 + l_mut_10;
const bool l_12 = l_11 >= uint(3);
if (l_12) {
const uint l_13 = l_4 + l_mut_10;
const uint l_14 = l_13 - uint(3);
const uint l_15 = l_7 + l_14;
const float l_16 = buffer_0[l_15];
const uint l_17 = l_8 + l_mut_10;
const float l_18 = buffer_1[l_17];
const float l_19 = l_16 * l_18;
l_mut_9 = l_mut_9 + l_19;
}
}
const uint l_20 = l_7 + l_4;
const float l_21 = -l_mut_9;
const float l_22 = exp(l_21);
const float l_23 = float(1.0) + l_22;
const float l_24 = l_mut_9 / l_23;
buffer_2[l_20] = l_24;
}

}
