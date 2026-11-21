#include <metal_stdlib>
using namespace metal;

struct info_st {
    uint static_meta[8];
};

[[kernel]]
void hadamard_fwht_f_(
    const device half * buffer_0 [[buffer(0)]],
    const device float * buffer_1 [[buffer(1)]],
    device half * buffer_2 [[buffer(2)]],
    const device uint * buffer_3 [[buffer(3)]],
    constant info_st& info [[buffer(4)]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint3 thread_pos_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_pos_in_grid [[threadgroup_position_in_grid]],
    uint simd_size [[threads_per_simdgroup]]
) {
uint threads_per_simdgroup_checked = min(simd_size, threads_per_threadgroup.x * threads_per_threadgroup.y * threads_per_threadgroup.z);
threadgroup uchar dynamic_shared_mem[512];
// Shared array size: 128, 512 bytes
threadgroup float* shared_memory_4 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[0]);
float l_arr_5[1];

float l_arr_6[1];

uint l_mut_7;
float l_mut_17;
float l_mut_25;
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
const uint l_10 = l_mut_7 * uint(128);
const uint l_11 = l_mut_7 % l_0;
const uint l_12 = l_11 * uint(128);
const uint l_13 = uint(0) + thread_pos_in_threadgroup.x;
const uint l_14 = l_10 + l_13;
const half l_15 = buffer_0[l_14];
const float l_16 = float(l_15);
l_mut_17 = l_16;
const bool l_18 = l_2 == uint(0);
const bool l_19 = l_1 != uint(0);
const bool l_20 = l_18 && l_19;
if (l_20) {
const uint l_21 = l_12 + l_13;
const float l_22 = buffer_1[l_21];
l_mut_17 = l_mut_17 * l_22;
}
const float l_23 = { simd_shuffle_xor(l_mut_17, uint(1)) };
const float l_24 = l_mut_17 + l_23;
l_mut_25 = l_24;
const uint l_26 = thread_pos_in_threadgroup.x & uint(1);
const bool l_27 = l_26 != uint(0);
if (l_27) {
const float l_28 = l_23 - l_mut_17;
l_mut_25 = l_28;
}
l_mut_17 = l_mut_25;
const float l_29 = { simd_shuffle_xor(l_mut_17, uint(2)) };
const float l_30 = l_mut_17 + l_29;
l_mut_25 = l_30;
const uint l_31 = thread_pos_in_threadgroup.x & uint(2);
const bool l_32 = l_31 != uint(0);
if (l_32) {
const float l_33 = l_29 - l_mut_17;
l_mut_25 = l_33;
}
l_mut_17 = l_mut_25;
const float l_34 = { simd_shuffle_xor(l_mut_17, uint(4)) };
const float l_35 = l_mut_17 + l_34;
l_mut_25 = l_35;
const uint l_36 = thread_pos_in_threadgroup.x & uint(4);
const bool l_37 = l_36 != uint(0);
if (l_37) {
const float l_38 = l_34 - l_mut_17;
l_mut_25 = l_38;
}
l_mut_17 = l_mut_25;
const float l_39 = { simd_shuffle_xor(l_mut_17, uint(8)) };
const float l_40 = l_mut_17 + l_39;
l_mut_25 = l_40;
const uint l_41 = thread_pos_in_threadgroup.x & uint(8);
const bool l_42 = l_41 != uint(0);
if (l_42) {
const float l_43 = l_39 - l_mut_17;
l_mut_25 = l_43;
}
l_mut_17 = l_mut_25;
const float l_44 = { simd_shuffle_xor(l_mut_17, uint(16)) };
const float l_45 = l_mut_17 + l_44;
l_mut_25 = l_45;
const uint l_46 = thread_pos_in_threadgroup.x & uint(16);
const bool l_47 = l_46 != uint(0);
if (l_47) {
const float l_48 = l_44 - l_mut_17;
l_mut_25 = l_48;
}
l_mut_17 = l_mut_25;
l_arr_5[uint(0)] = l_mut_17;
const uint l_49 = uint(0) + thread_pos_in_threadgroup.x;
const float l_50 = l_arr_5[uint(0)];
shared_memory_4[l_49] = l_50;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_51 = thread_pos_in_threadgroup.x ^ uint(32);
const uint l_52 = thread_pos_in_threadgroup.x & uint(32);
const bool l_53 = l_52 != uint(0);
const float l_54 = l_arr_5[uint(0)];
const uint l_55 = uint(0) + l_51;
const float l_56 = shared_memory_4[l_55];
const float l_57 = l_54 + l_56;
l_mut_25 = l_57;
if (l_53) {
const float l_58 = l_56 - l_54;
l_mut_25 = l_58;
}
l_arr_6[uint(0)] = l_mut_25;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float l_59 = l_arr_6[uint(0)];
l_arr_5[uint(0)] = l_59;
const uint l_60 = uint(0) + thread_pos_in_threadgroup.x;
shared_memory_4[l_60] = l_59;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_61 = thread_pos_in_threadgroup.x ^ uint(64);
const uint l_62 = thread_pos_in_threadgroup.x & uint(64);
const bool l_63 = l_62 != uint(0);
const float l_64 = l_arr_5[uint(0)];
const uint l_65 = uint(0) + l_61;
const float l_66 = shared_memory_4[l_65];
const float l_67 = l_64 + l_66;
l_mut_25 = l_67;
if (l_63) {
const float l_68 = l_66 - l_64;
l_mut_25 = l_68;
}
l_arr_6[uint(0)] = l_mut_25;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float l_69 = l_arr_6[uint(0)];
l_arr_5[uint(0)] = l_69;
const uint l_70 = uint(0) + thread_pos_in_threadgroup.x;
const float l_71 = l_arr_5[uint(0)];
const float l_72 = l_71 * float(0.0883883461356163);
l_mut_25 = l_72;
const bool l_73 = l_2 != uint(0);
const bool l_74 = l_1 != uint(0);
const bool l_75 = l_73 && l_74;
if (l_75) {
const uint l_76 = l_12 + l_70;
const float l_77 = buffer_1[l_76];
l_mut_25 = l_mut_25 * l_77;
}
const uint l_78 = l_10 + l_70;
const half l_79 = half(l_mut_25);
buffer_2[l_78] = l_79;
l_mut_7 = l_mut_7 + threadgroups_per_grid.x;
}

}
