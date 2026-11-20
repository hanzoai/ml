#include <metal_stdlib>
using namespace metal;

struct info_st {
    uint static_meta[14];
};

[[kernel]]
void gdn_gating_io_f_(
    const device bfloat * buffer_0 [[buffer(0)]],
    const device bfloat * buffer_1 [[buffer(1)]],
    const device float * buffer_2 [[buffer(2)]],
    const device float * buffer_3 [[buffer(3)]],
    device bfloat * buffer_4 [[buffer(4)]],
    device bfloat * buffer_5 [[buffer(5)]],
    const device uint * buffer_6 [[buffer(6)]],
    constant info_st& info [[buffer(7)]],
    uint3 thread_pos_in_grid [[thread_position_in_grid]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]]
) {
uint thread_index_in_grid = (
                thread_pos_in_grid.z * threadgroups_per_grid.x * threads_per_threadgroup.x * threadgroups_per_grid.y * threads_per_threadgroup.y)
                + (thread_pos_in_grid.y * threadgroups_per_grid.x * threads_per_threadgroup.x)
                + thread_pos_in_grid.x;
float l_mut_11;
const uint l_0 = buffer_6[uint(0)];
const bool l_1 = thread_index_in_grid < l_0;
if (l_1) {
const uint l_2 = buffer_6[uint(1)];
const uint l_3 = thread_index_in_grid % l_2;
const bfloat l_4 = buffer_0[thread_index_in_grid];
const float l_5 = float(l_4);
const bfloat l_6 = buffer_1[thread_index_in_grid];
const float l_7 = float(l_6);
const float l_8 = buffer_3[l_3];
const float l_9 = l_7 + l_8;
const bool l_10 = l_9 > float(20.0);
if (l_10) {
l_mut_11 = l_9;
} else {
const float l_12 = exp(l_9);
const float l_13 = float(1.0) + l_12;
const float l_14 = log(l_13);
l_mut_11 = l_14;
}
const float l_15 = -l_5;
const float l_16 = exp(l_15);
const float l_17 = float(1.0) + l_16;
const float l_18 = float(1.0) / l_17;
const float l_19 = buffer_2[l_3];
const float l_20 = exp(l_19);
const float l_21 = -l_20;
const float l_22 = l_21 * l_mut_11;
const bfloat l_23 = bfloat(l_18);
buffer_4[thread_index_in_grid] = l_23;
const bfloat l_24 = bfloat(l_22);
buffer_5[thread_index_in_grid] = l_24;
}

}
