#include <metal_stdlib>
using namespace metal;

struct info_st {
    uint static_meta[8];
};

[[kernel]]
void hadamard_fwht_f_f32(
    const device float * buffer_0 [[buffer(0)]],
    const device float * buffer_1 [[buffer(1)]],
    device float * buffer_2 [[buffer(2)]],
    const device uint * buffer_3 [[buffer(3)]],
    constant info_st& info [[buffer(4)]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint3 thread_pos_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_pos_in_grid [[threadgroup_position_in_grid]],
    uint simd_size [[threads_per_simdgroup]]
) {
uint threads_per_simdgroup_checked = min(simd_size, threads_per_threadgroup.x * threads_per_threadgroup.y * threads_per_threadgroup.z);
threadgroup uchar dynamic_shared_mem[1024];
// Shared array size: 256, 1024 bytes
threadgroup float* shared_memory_4 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[0]);
float l_arr_5[1];

float l_arr_6[1];

uint l_mut_7;
float l_mut_16;
float l_mut_24;
const uint l_0 = buffer_3[uint(1)];
const uint l_1 = buffer_3[uint(2)];
const uint l_2 = buffer_3[uint(3)];
const uint l_3 = buffer_3[uint(4)];
l_mut_7 = threadgroup_pos_in_grid.x;
while (true) {
const bool l_8 = l_mut_7 < l_3;
const bool l_9 = !l_8;
if (l_9) {
break;}
const uint l_10 = l_mut_7 * uint(256);
const uint l_11 = l_mut_7 % l_0;
const uint l_12 = l_11 * uint(256);
const uint l_13 = uint(0) + thread_pos_in_threadgroup.x;
const uint l_14 = l_10 + l_13;
const float l_15 = buffer_0[l_14];
l_mut_16 = l_15;
const bool l_17 = l_2 == uint(0);
const bool l_18 = l_1 != uint(0);
const bool l_19 = l_17 && l_18;
if (l_19) {
const uint l_20 = l_12 + l_13;
const float l_21 = buffer_1[l_20];
l_mut_16 = l_mut_16 * l_21;
}
const float l_22 = { simd_shuffle_xor(l_mut_16, uint(1)) };
const float l_23 = l_mut_16 + l_22;
l_mut_24 = l_23;
const uint l_25 = thread_pos_in_threadgroup.x & uint(1);
const bool l_26 = l_25 != uint(0);
if (l_26) {
const float l_27 = l_22 - l_mut_16;
l_mut_24 = l_27;
}
l_mut_16 = l_mut_24;
const float l_28 = { simd_shuffle_xor(l_mut_16, uint(2)) };
const float l_29 = l_mut_16 + l_28;
l_mut_24 = l_29;
const uint l_30 = thread_pos_in_threadgroup.x & uint(2);
const bool l_31 = l_30 != uint(0);
if (l_31) {
const float l_32 = l_28 - l_mut_16;
l_mut_24 = l_32;
}
l_mut_16 = l_mut_24;
const float l_33 = { simd_shuffle_xor(l_mut_16, uint(4)) };
const float l_34 = l_mut_16 + l_33;
l_mut_24 = l_34;
const uint l_35 = thread_pos_in_threadgroup.x & uint(4);
const bool l_36 = l_35 != uint(0);
if (l_36) {
const float l_37 = l_33 - l_mut_16;
l_mut_24 = l_37;
}
l_mut_16 = l_mut_24;
const float l_38 = { simd_shuffle_xor(l_mut_16, uint(8)) };
const float l_39 = l_mut_16 + l_38;
l_mut_24 = l_39;
const uint l_40 = thread_pos_in_threadgroup.x & uint(8);
const bool l_41 = l_40 != uint(0);
if (l_41) {
const float l_42 = l_38 - l_mut_16;
l_mut_24 = l_42;
}
l_mut_16 = l_mut_24;
const float l_43 = { simd_shuffle_xor(l_mut_16, uint(16)) };
const float l_44 = l_mut_16 + l_43;
l_mut_24 = l_44;
const uint l_45 = thread_pos_in_threadgroup.x & uint(16);
const bool l_46 = l_45 != uint(0);
if (l_46) {
const float l_47 = l_43 - l_mut_16;
l_mut_24 = l_47;
}
l_mut_16 = l_mut_24;
l_arr_5[uint(0)] = l_mut_16;
const uint l_48 = uint(0) + thread_pos_in_threadgroup.x;
const float l_49 = l_arr_5[uint(0)];
shared_memory_4[l_48] = l_49;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_50 = thread_pos_in_threadgroup.x ^ uint(32);
const uint l_51 = thread_pos_in_threadgroup.x & uint(32);
const bool l_52 = l_51 != uint(0);
const float l_53 = l_arr_5[uint(0)];
const uint l_54 = uint(0) + l_50;
const float l_55 = shared_memory_4[l_54];
const float l_56 = l_53 + l_55;
l_mut_24 = l_56;
if (l_52) {
const float l_57 = l_55 - l_53;
l_mut_24 = l_57;
}
l_arr_6[uint(0)] = l_mut_24;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float l_58 = l_arr_6[uint(0)];
l_arr_5[uint(0)] = l_58;
const uint l_59 = uint(0) + thread_pos_in_threadgroup.x;
shared_memory_4[l_59] = l_58;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_60 = thread_pos_in_threadgroup.x ^ uint(64);
const uint l_61 = thread_pos_in_threadgroup.x & uint(64);
const bool l_62 = l_61 != uint(0);
const float l_63 = l_arr_5[uint(0)];
const uint l_64 = uint(0) + l_60;
const float l_65 = shared_memory_4[l_64];
const float l_66 = l_63 + l_65;
l_mut_24 = l_66;
if (l_62) {
const float l_67 = l_65 - l_63;
l_mut_24 = l_67;
}
l_arr_6[uint(0)] = l_mut_24;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float l_68 = l_arr_6[uint(0)];
l_arr_5[uint(0)] = l_68;
const uint l_69 = uint(0) + thread_pos_in_threadgroup.x;
shared_memory_4[l_69] = l_68;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_70 = thread_pos_in_threadgroup.x ^ uint(128);
const uint l_71 = thread_pos_in_threadgroup.x & uint(128);
const bool l_72 = l_71 != uint(0);
const float l_73 = l_arr_5[uint(0)];
const uint l_74 = uint(0) + l_70;
const float l_75 = shared_memory_4[l_74];
const float l_76 = l_73 + l_75;
l_mut_24 = l_76;
if (l_72) {
const float l_77 = l_75 - l_73;
l_mut_24 = l_77;
}
l_arr_6[uint(0)] = l_mut_24;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float l_78 = l_arr_6[uint(0)];
l_arr_5[uint(0)] = l_78;
const uint l_79 = uint(0) + thread_pos_in_threadgroup.x;
const float l_80 = l_arr_5[uint(0)];
const float l_81 = l_80 * float(0.0625);
l_mut_24 = l_81;
const bool l_82 = l_2 != uint(0);
const bool l_83 = l_1 != uint(0);
const bool l_84 = l_82 && l_83;
if (l_84) {
const uint l_85 = l_12 + l_79;
const float l_86 = buffer_1[l_85];
l_mut_24 = l_mut_24 * l_86;
}
const uint l_87 = l_10 + l_79;
buffer_2[l_87] = l_mut_24;
l_mut_7 = l_mut_7 + threadgroups_per_grid.x;
}

}
