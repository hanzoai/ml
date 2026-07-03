using namespace metal;

struct info_st {
    uint static_meta[10];
};

[[kernel]]
void rms_norm_blk_f_f32(
    const device float * buffer_0 [[buffer(0)]],
    const device float * buffer_1 [[buffer(1)]],
    device float * buffer_2 [[buffer(2)]],
    const device float * buffer_3 [[buffer(3)]],
    const device uint * buffer_4 [[buffer(4)]],
    constant info_st& info [[buffer(5)]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 thread_pos_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_pos_in_grid [[threadgroup_position_in_grid]]
) {
uint total_thread_in_threadgroup = threads_per_threadgroup.x * threads_per_threadgroup.y * threads_per_threadgroup.z;
uint threadgroup_index_in_grid = (threadgroup_pos_in_grid.z * threadgroups_per_grid.y * threadgroups_per_grid.x) + (threadgroup_pos_in_grid.y * threadgroups_per_grid.x) + threadgroup_pos_in_grid.x;
threadgroup uchar dynamic_shared_mem[512];
// Shared array size: 128, 512 bytes
threadgroup float* shared_memory_9 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[0]);
uint l_mut_3;
uint l_mut_11;
uint l_mut_25;
float l_mut_2;
const uint l_0 = buffer_4[uint(0)];
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
const float l_8 = l_7 * l_7;
l_mut_2 = l_mut_2 + l_8;
l_mut_3 = l_mut_3 + total_thread_in_threadgroup;
}
shared_memory_9[thread_index_in_threadgroup] = l_mut_2;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_10 = total_thread_in_threadgroup / uint(2);
l_mut_11 = l_10;
while (true) {
const bool l_12 = l_mut_11 > uint(0);
const bool l_13 = !l_12;
if (l_13) {
break;}
const bool l_14 = thread_index_in_threadgroup < l_mut_11;
if (l_14) {
const uint l_15 = thread_index_in_threadgroup + l_mut_11;
const float l_16 = shared_memory_9[l_15];
const float l_17 = shared_memory_9[thread_index_in_threadgroup];
const float l_18 = l_17 + l_16;
shared_memory_9[thread_index_in_threadgroup] = l_18;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
l_mut_11 = l_mut_11 / uint(2);
}
const float l_19 = shared_memory_9[uint(0)];
const float l_20 = float(l_0);
const float l_21 = l_19 / l_20;
const float l_22 = buffer_3[uint(0)];
const float l_23 = l_21 + l_22;
const float l_24 = sqrt(l_23);
l_mut_25 = thread_index_in_threadgroup;
while (true) {
const bool l_26 = l_mut_25 < l_0;
const bool l_27 = !l_26;
if (l_27) {
break;}
const uint l_28 = l_1 + l_mut_25;
const uint l_29 = l_1 + l_mut_25;
const float l_30 = buffer_0[l_29];
const float l_31 = l_30 / l_24;
const float l_32 = buffer_1[l_mut_25];
const float l_33 = l_31 * l_32;
buffer_2[l_28] = l_33;
l_mut_25 = l_mut_25 + total_thread_in_threadgroup;
}

}
