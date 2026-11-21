#include <metal_stdlib>
using namespace metal;

struct info_st {
    uint static_meta[12];
};

[[kernel]]
void reshape_and_cache_f_(
    const device bfloat * buffer_0 [[buffer(0)]],
    const device bfloat * buffer_1 [[buffer(1)]],
    device bfloat * buffer_2 [[buffer(2)]],
    device bfloat * buffer_3 [[buffer(3)]],
    const device long * buffer_4 [[buffer(4)]],
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
const uint l_0 = buffer_5[uint(0)];
const uint l_1 = buffer_5[uint(1)];
const uint l_2 = buffer_5[uint(2)];
const uint l_3 = buffer_5[uint(3)];
const uint l_4 = l_1 * l_2;
const uint l_5 = l_0 * l_4;
const bool l_6 = thread_index_in_grid < l_5;
if (l_6) {
const uint l_7 = thread_index_in_grid / l_4;
const long l_8 = buffer_4[l_7];
const bool l_9 = l_8 >= long(0);
if (l_9) {
const uint l_10 = l_7 * l_4;
const uint l_11 = thread_index_in_grid - l_10;
const uint l_12 = l_11 / l_2;
const uint l_13 = l_12 * l_2;
const uint l_14 = l_11 - l_13;
const uint l_15 = uint(l_8);
const uint l_16 = l_15 / l_3;
const uint l_17 = l_15 % l_3;
const uint l_18 = l_14 / uint(8);
const uint l_19 = l_14 % uint(8);
const uint l_20 = l_16 * l_1;
const uint l_21 = l_20 + l_12;
const uint l_22 = l_2 / uint(8);
const uint l_23 = l_21 * l_22;
const uint l_24 = l_23 + l_18;
const uint l_25 = l_24 * l_3;
const uint l_26 = l_25 + l_17;
const uint l_27 = l_26 * uint(8);
const uint l_28 = l_27 + l_19;
const uint l_29 = l_16 * l_1;
const uint l_30 = l_29 + l_12;
const uint l_31 = l_30 * l_2;
const uint l_32 = l_31 + l_14;
const uint l_33 = l_32 * l_3;
const uint l_34 = l_33 + l_17;
const bfloat l_35 = buffer_0[thread_index_in_grid];
buffer_2[l_28] = l_35;
const bfloat l_36 = buffer_1[thread_index_in_grid];
buffer_3[l_34] = l_36;
}
}

}
