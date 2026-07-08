// GENERATED — DO NOT EDIT. Lowered from hanzo-kernel DSL `norm::rms_norm_blk (f32 + f16)` (hanzo-kernel/src/norm.rs) to HIP
// via cubecl-hip at target=Rocm, nt=1024 (island 'rocm' arm = plane_sum warp-shuffle block reduction,
// single cross-warp barrier -- matching the incumbent's __shfl reduction). The DSL Rust is authoritative;
// ml compiles+launches this via its existing hipcc pipeline. Regenerate: rms_norm_blk_rocm_bit_exact with CUBECL_DEBUG_LOG.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
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
    uint32 static_meta[10];
};


// ---- f32 I/O instantiation ----
extern "C" __global__ void __launch_bounds__(1024) rms_norm_blk_f_f32(
    const float* __restrict__ buffer_0, const float* __restrict__ buffer_1, float* buffer_2, const float* __restrict__ buffer_3, const uint32* __restrict__ buffer_4, const info_st* __restrict__ info_ptr
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
float *shared_memory_9 = reinterpret_cast<float*>(&dynamic_shared_mem[0]);
float l_mut_2;
float l_mut_16;
uint32 l_mut_3;
uint32 l_mut_25;
const uint32 l_0 = buffer_4[uint32(0)];
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
const float l_8 = l_7 * l_7;
l_mut_2 = l_mut_2 + l_8;
l_mut_3 = l_mut_3 + blockDimGlobal;
}
auto plane_l_10 = [&]() -> float {
    float acc = l_mut_2;
    for (uint offset = 1; offset < warpSizeChecked; offset *=2 ) {
acc += __shfl_xor(acc, offset);
    };
    return acc;
};
const float l_10 = plane_l_10();
const uint32 l_11 = blockDimGlobal / warpSize;
const uint32 l_12 = threadIdxGlobal % warpSize;
const bool l_13 = l_12 == uint32(0);
if (l_13) {
const uint32 l_14 = threadIdxGlobal / warpSize;
shared_memory_9[l_14] = l_10;
}
__syncthreads();

const uint32 l_15 = threadIdxGlobal % warpSize;
l_mut_16 = float(0.0);
const bool l_17 = l_15 < l_11;
if (l_17) {
const float l_18 = shared_memory_9[l_15];
l_mut_16 = l_18;
}
auto plane_l_19 = [&]() -> float {
    float acc = l_mut_16;
    for (uint offset = 1; offset < warpSizeChecked; offset *=2 ) {
acc += __shfl_xor(acc, offset);
    };
    return acc;
};
const float l_19 = plane_l_19();
const float l_20 = float(l_0);
const float l_21 = l_19 / l_20;
const float l_22 = buffer_3[uint32(0)];
const float l_23 = l_21 + l_22;
const float l_24 = sqrt(l_23);
l_mut_25 = threadIdxGlobal;
while (true) {
const bool l_26 = l_mut_25 < l_0;
const bool l_27 = !l_26;
if (l_27) {
break;}
const uint32 l_28 = l_1 + l_mut_25;
const uint32 l_29 = l_1 + l_mut_25;
const float l_30 = buffer_0[l_29];
const float l_31 = l_30 / l_24;
const float l_32 = buffer_1[l_mut_25];
const float l_33 = l_31 * l_32;
buffer_2[l_28] = l_33;
l_mut_25 = l_mut_25 + blockDimGlobal;
}

}

// ---- f16 I/O instantiation (shares the info_st preamble above) ----
extern "C" __global__ void __launch_bounds__(1024) rms_norm_blk_f_(
    const __half* __restrict__ buffer_0, const __half* __restrict__ buffer_1, __half* buffer_2, const float* __restrict__ buffer_3, const uint32* __restrict__ buffer_4, const info_st* __restrict__ info_ptr
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
float *shared_memory_10 = reinterpret_cast<float*>(&dynamic_shared_mem[0]);
uint32 l_mut_3;
uint32 l_mut_26;
float l_mut_2;
float l_mut_17;
const uint32 l_0 = buffer_4[uint32(0)];
const uint32 l_1 = blockIdxGlobal * l_0;
l_mut_2 = float(0.0);
l_mut_3 = threadIdxGlobal;
while (true) {
const bool l_4 = l_mut_3 < l_0;
const bool l_5 = !l_4;
if (l_5) {
break;}
const uint32 l_6 = l_1 + l_mut_3;
const __half l_7 = buffer_0[l_6];
const float l_8 = float(l_7);
const float l_9 = l_8 * l_8;
l_mut_2 = l_mut_2 + l_9;
l_mut_3 = l_mut_3 + blockDimGlobal;
}
auto plane_l_11 = [&]() -> float {
    float acc = l_mut_2;
    for (uint offset = 1; offset < warpSizeChecked; offset *=2 ) {
acc += __shfl_xor(acc, offset);
    };
    return acc;
};
const float l_11 = plane_l_11();
const uint32 l_12 = blockDimGlobal / warpSize;
const uint32 l_13 = threadIdxGlobal % warpSize;
const bool l_14 = l_13 == uint32(0);
if (l_14) {
const uint32 l_15 = threadIdxGlobal / warpSize;
shared_memory_10[l_15] = l_11;
}
__syncthreads();

const uint32 l_16 = threadIdxGlobal % warpSize;
l_mut_17 = float(0.0);
const bool l_18 = l_16 < l_12;
if (l_18) {
const float l_19 = shared_memory_10[l_16];
l_mut_17 = l_19;
}
auto plane_l_20 = [&]() -> float {
    float acc = l_mut_17;
    for (uint offset = 1; offset < warpSizeChecked; offset *=2 ) {
acc += __shfl_xor(acc, offset);
    };
    return acc;
};
const float l_20 = plane_l_20();
const float l_21 = float(l_0);
const float l_22 = l_20 / l_21;
const float l_23 = buffer_3[uint32(0)];
const float l_24 = l_22 + l_23;
const float l_25 = sqrt(l_24);
l_mut_26 = threadIdxGlobal;
while (true) {
const bool l_27 = l_mut_26 < l_0;
const bool l_28 = !l_27;
if (l_28) {
break;}
const uint32 l_29 = l_1 + l_mut_26;
const uint32 l_30 = l_1 + l_mut_26;
const __half l_31 = buffer_0[l_30];
const float l_32 = float(l_31);
const float l_33 = l_32 / l_25;
const __half l_34 = buffer_1[l_mut_26];
const float l_35 = float(l_34);
const float l_36 = l_33 * l_35;
const __half l_37 = __half(l_36);
buffer_2[l_29] = l_37;
l_mut_26 = l_mut_26 + blockDimGlobal;
}

}
