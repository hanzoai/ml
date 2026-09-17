#include <metal_stdlib>
using namespace metal;

struct info_st {
    uint static_meta[8];
};

[[kernel]]
void gemv_f_f32(
    const device float * buffer_0 [[buffer(0)]],
    const device float * buffer_1 [[buffer(1)]],
    device float * buffer_2 [[buffer(2)]],
    const device uint * buffer_3 [[buffer(3)]],
    constant info_st& info [[buffer(4)]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 thread_pos_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_pos_in_grid [[threadgroup_position_in_grid]]
) {
uint total_thread_in_threadgroup = threads_per_threadgroup.x * threads_per_threadgroup.y * threads_per_threadgroup.z;
uint threadgroup_index_in_grid = (threadgroup_pos_in_grid.z * threadgroups_per_grid.y * threadgroups_per_grid.x) + (threadgroup_pos_in_grid.y * threadgroups_per_grid.x) + threadgroup_pos_in_grid.x;
threadgroup uchar dynamic_shared_mem[128];
// Shared array size: 32, 128 bytes
threadgroup float* shared_memory_10 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[0]);
float l_mut_2;
uint l_mut_3;
uint l_mut_12;
const uint l_0 = buffer_3[uint(0)];
const uint l_1 = threadgroup_index_in_grid * l_0;
l_mut_2 = float(0.0);
l_mut_3 = thread_index_in_threadgroup;
while (true) {
const bool l_4 = l_mut_3 < l_0;
const bool l_5 = !l_4;
if (l_5) {
break;}
const uint l_6 = l_1 + l_mut_3;
const float l_7 = buffer_0[l_6];
const float l_8 = buffer_1[l_mut_3];
const float l_9 = l_7 * l_8;
l_mut_2 = l_mut_2 + l_9;
l_mut_3 = l_mut_3 + uint(32);
}
shared_memory_10[thread_index_in_threadgroup] = l_mut_2;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_11 = total_thread_in_threadgroup / uint(2);
l_mut_12 = l_11;
while (true) {
const bool l_13 = l_mut_12 > uint(0);
const bool l_14 = !l_13;
if (l_14) {
break;}
const bool l_15 = thread_index_in_threadgroup < l_mut_12;
if (l_15) {
const uint l_16 = thread_index_in_threadgroup + l_mut_12;
const float l_17 = shared_memory_10[l_16];
const float l_18 = shared_memory_10[thread_index_in_threadgroup];
const float l_19 = l_18 + l_17;
shared_memory_10[thread_index_in_threadgroup] = l_19;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
l_mut_12 = l_mut_12 / uint(2);
}
const bool l_20 = thread_index_in_threadgroup == uint(0);
if (l_20) {
const float l_21 = shared_memory_10[uint(0)];
buffer_2[threadgroup_index_in_grid] = l_21;
}

}
