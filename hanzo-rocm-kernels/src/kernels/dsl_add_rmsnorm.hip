// GENERATED — DO NOT EDIT. Lowered from hanzo-kernel DSL `norm::add_rmsnorm_blk (f32)` (hanzo-kernel/src/norm.rs) to HIP
// via cubecl-hip at target=Rocm, nt=1024 (island 'rocm' arm = plane_sum warp-shuffle block reduction,
// single cross-warp barrier -- matching the incumbent's __shfl reduction). The DSL Rust is authoritative;
// ml compiles+launches this via its existing hipcc pipeline. Regenerate: add_rmsnorm_blk_rocm_bit_exact with CUBECL_DEBUG_LOG.

#include <hip/hip_runtime.h>
typedef unsigned int uint;
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long int uint64;
typedef signed char int8;
typedef signed short int16;
typedef signed int int32;
typedef signed long long int int64;

struct info_st {
    uint32 static_meta[14];
};


extern "C" __global__ void __launch_bounds__(1024) add_rmsnorm_blk_f_f32(
    const float* __restrict__ buffer_0, const float* __restrict__ buffer_1, const float* __restrict__ buffer_2, float* buffer_3, float* buffer_4, const float* __restrict__ buffer_5, const uint32* __restrict__ buffer_6, const info_st* __restrict__ info_ptr
) {
uint32 threadIdxGlobal = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
uint32 blockDimGlobal = blockDim.x * blockDim.y * blockDim.z;
uint32 blockIdxGlobal = (blockIdx.z * gridDim.y * gridDim.x) + (blockIdx.y * gridDim.x) + blockIdx.x;
uint32 warpSizeChecked = min(warpSize, blockDim.x * blockDim.y * blockDim.z);
extern __shared__ __align__(4) uchar dynamic_shared_mem[];
const info_st& info = *info_ptr;
const uint32* dynamic_meta = reinterpret_cast<const uint32*>(
                    reinterpret_cast<const char*>(info_ptr) + sizeof(info_st)
                );

// Shared array size: 1024, 4096 bytes
float *shared_memory_13 = reinterpret_cast<float*>(&dynamic_shared_mem[0]);
uint32 l_mut_3;
uint32 l_mut_30;
float l_mut_2;
float l_mut_19;
const uint32 l_0 = buffer_6[uint32(0)];
const uint32 l_1 = blockIdxGlobal * l_0;
l_mut_2 = float(0.0);
l_mut_3 = threadIdxGlobal;
while (true) {
const bool l_4 = l_mut_3 < l_0;
const bool l_5 = !l_4;
if (l_5) {
break;}
const uint32 l_6 = l_1 + l_mut_3;
const float l_7 = buffer_0[l_6];
const uint32 l_8 = l_1 + l_mut_3;
const float l_9 = buffer_1[l_8];
const float l_10 = l_7 + l_9;
const uint32 l_11 = l_1 + l_mut_3;
buffer_3[l_11] = l_10;
const float l_12 = l_10 * l_10;
l_mut_2 = l_mut_2 + l_12;
l_mut_3 = l_mut_3 + blockDimGlobal;
}
auto plane_l_14 = [&]() -> float {
    float acc = l_mut_2;
    for (uint offset = 1; offset < warpSizeChecked; offset *=2 ) {
acc += __shfl_xor(acc, offset);
    };
    return acc;
};
const float l_14 = plane_l_14();
const uint32 l_15 = blockDimGlobal / warpSize;
const uint32 l_16 = threadIdxGlobal % warpSize;
const bool l_17 = l_16 == uint32(0);
if (l_17) {
const uint32 l_18 = threadIdxGlobal / warpSize;
shared_memory_13[l_18] = l_14;
}
__syncthreads();

l_mut_19 = float(0.0);
const bool l_20 = threadIdxGlobal < l_15;
if (l_20) {
const float l_21 = shared_memory_13[threadIdxGlobal];
l_mut_19 = l_21;
}
auto plane_l_22 = [&]() -> float {
    float acc = l_mut_19;
    for (uint offset = 1; offset < warpSizeChecked; offset *=2 ) {
acc += __shfl_xor(acc, offset);
    };
    return acc;
};
const float l_22 = plane_l_22();
const bool l_23 = threadIdxGlobal == uint32(0);
if (l_23) {
shared_memory_13[uint32(0)] = l_22;
}
__syncthreads();

const float l_24 = shared_memory_13[uint32(0)];
const float l_25 = float(l_0);
const float l_26 = l_24 / l_25;
const float l_27 = buffer_5[uint32(0)];
const float l_28 = l_26 + l_27;
const float l_29 = sqrt(l_28);
l_mut_30 = threadIdxGlobal;
while (true) {
const bool l_31 = l_mut_30 < l_0;
const bool l_32 = !l_31;
if (l_32) {
break;}
const uint32 l_33 = l_1 + l_mut_30;
const float l_34 = buffer_0[l_33];
const uint32 l_35 = l_1 + l_mut_30;
const float l_36 = buffer_1[l_35];
const float l_37 = l_34 + l_36;
const uint32 l_38 = l_1 + l_mut_30;
const float l_39 = l_37 / l_29;
const float l_40 = buffer_2[l_mut_30];
const float l_41 = l_39 * l_40;
buffer_4[l_38] = l_41;
l_mut_30 = l_mut_30 + blockDimGlobal;
}

}
