#include <metal_stdlib>
using namespace metal;

struct info_st {
    uint static_meta[14];
};

[[kernel]]
void gdn_gating_f_(
    const device half * buffer_0 [[buffer(0)]],
    const device half * buffer_1 [[buffer(1)]],
    const device float * buffer_2 [[buffer(2)]],
    const device float * buffer_3 [[buffer(3)]],
    device half * buffer_4 [[buffer(4)]],
    device half * buffer_5 [[buffer(5)]],
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
float l_mut_15;
float l_mut_21;
const uint l_0 = info.static_meta[uint(11)];
const bool l_1 = thread_index_in_grid < l_0;
if (l_1) {
const uint l_2 = buffer_6[uint(0)];
const uint l_3 = buffer_6[uint(1)];
const uint l_4 = thread_index_in_grid % l_2;
const half l_5 = buffer_0[thread_index_in_grid];
const float l_6 = float(l_5);
const half l_7 = buffer_1[thread_index_in_grid];
const float l_8 = float(l_7);
const float l_9 = -l_6;
const float l_10 = exp(l_9);
const float l_11 = float(1.0) + l_10;
const float l_12 = float(1.0) / l_11;
const float l_13 = buffer_3[l_4];
const float l_14 = l_8 + l_13;
l_mut_15 = l_14;
const bool l_16 = l_14 <= float(20.0);
if (l_16) {
const float l_17 = exp(l_14);
const float l_18 = float(1.0) + l_17;
const float l_19 = log(l_18);
l_mut_15 = l_19;
}
const float l_20 = buffer_2[l_4];
l_mut_21 = l_20;
const bool l_22 = l_3 == uint(0);
if (l_22) {
const float l_23 = exp(l_mut_21);
const float l_24 = -l_23;
l_mut_21 = l_24;
}
const float l_25 = l_mut_21 * l_mut_15;
const half l_26 = half(l_12);
buffer_4[thread_index_in_grid] = l_26;
const half l_27 = half(l_25);
buffer_5[thread_index_in_grid] = l_27;
}

}
