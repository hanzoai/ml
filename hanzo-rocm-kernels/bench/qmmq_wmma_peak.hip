// Raw WMMA issue-rate microbench: hammer iu8 vs f16 WMMA from registers only (no LDS, no global) to
// find the TRUE matrix-core ceiling on gfx1151. Each thread runs a long dependent-but-pipelined chain
// of WMMA so the SIMD issues back-to-back; we measure TFLOP/s and compare iu8 vs f16.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <chrono>
#include <cstdio>
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

__global__ void peak_iu8(int* out,int reps){
    v4i a={0x01010101,0x01010101,0x01010101,0x01010101}, b=a;
    v8i c0={0,0,0,0,0,0,0,0},c1=c0,c2=c0,c3=c0;
    for(int i=0;i<reps;i++){
        c0=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,a,true,b,c0,false);
        c1=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,a,true,b,c1,false);
        c2=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,a,true,b,c2,false);
        c3=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,a,true,b,c3,false);
    }
    if(out) out[threadIdx.x]=c0[0]+c1[0]+c2[0]+c3[0];
}
__global__ void peak_f16(float* out,int reps){
    v16h a; for(int j=0;j<16;j++)a[j]=(_Float16)1.0f; v16h b=a;
    v8f c0={0,0,0,0,0,0,0,0},c1=c0,c2=c0,c3=c0;
    for(int i=0;i<reps;i++){
        c0=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a,b,c0);
        c1=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a,b,c1);
        c2=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a,b,c2);
        c3=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a,b,c3);
    }
    if(out) out[threadIdx.x]=c0[0]+c1[0]+c2[0]+c3[0];
}
int main(){
    int blocks=40*8, threads=256, reps=20000; // saturate all CUs
    double mac=2.0*16*16*16; // flops per 16x16x16 wmma
    double total_wmma=(double)blocks*threads/32.0*reps*4; // waves*reps*4 wmma
    auto run=[&](const char*nm,void(*k)(void*,int)){ // not used
    };
    // iu8
    hipDeviceSynchronize();
    for(int w=0;w<3;w++){ peak_iu8<<<blocks,threads>>>(nullptr,reps); } hipDeviceSynchronize();
    auto t0=std::chrono::high_resolution_clock::now();
    peak_iu8<<<blocks,threads>>>(nullptr,reps); hipDeviceSynchronize();
    auto t1=std::chrono::high_resolution_clock::now();
    double si=std::chrono::duration<double>(t1-t0).count();
    for(int w=0;w<3;w++){ peak_f16<<<blocks,threads>>>(nullptr,reps); } hipDeviceSynchronize();
    auto t2=std::chrono::high_resolution_clock::now();
    peak_f16<<<blocks,threads>>>(nullptr,reps); hipDeviceSynchronize();
    auto t3=std::chrono::high_resolution_clock::now();
    double sf=std::chrono::duration<double>(t3-t2).count();
    printf("iu8 WMMA peak: %.1f TOP/s  (%.3f ms)\n", total_wmma*mac/si/1e12, si*1e3);
    printf("f16 WMMA peak: %.1f TF/s   (%.3f ms)\n", total_wmma*mac/sf/1e12, sf*1e3);
    printf("iu8/f16 issue ratio: %.2fx\n", sf/si);
    return 0;
}
