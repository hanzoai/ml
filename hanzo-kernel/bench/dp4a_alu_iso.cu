// dp4a ALU-isolation microbench for the DSL CUDA matvec lane (GB10 sm_121, CUDA 13).
//
// BUILD/RUN (raw nvcc, not part of the cargo build):
//   nvcc -O3 -arch=sm_121 -o /tmp/dp4a_alu_iso hanzo-kernel/bench/dp4a_alu_iso.cu
//   /tmp/dp4a_alu_iso 8192 8192 200        # rows k iters ; 8192^2 int8 = 67 MB >> L2 -> cold
// Board numbers live in the commit message + CI, never here (a comment's number lies the moment the
// clock/power state changes -- and on this shared Grace-Blackwell SoC it changes run to run).
//
// PURPOSE: answer ONE question the DSL board leaves open -- does emitting a real __dp4a
// (instead of the 4 scalar IMADs cubecl-cpp 0.10 `Dot::format` actually emits for `Vector<i32,4>.dot`)
// change the DSL int8 decode matvec's time? The A/B holds the memory access pattern EXACTLY equal to
// the shipped DSL kernels (hanzo-kernel/src/quant.rs matvec_q8_dp4a_i8 and matvec_q8_dp4a_blk) and
// flips ONLY the inner dot idiom. Same bytes moved; the sole variable is ALU.
//
// Faithfulness to the DSL:
//   * i8pack  : 1 thread / row, block=64, thread `row` reads wq[row*ng + g] for g in 0..ng
//               (adjacent threads a full row apart -> uncoalesced)                [matvec_q8_dp4a_i8]
//   * blk     : 1 block / row, nt threads, thread t does g = j*nt+t for j in 0..ng/nt
//               (adjacent threads read adjacent Vector<i8,4> -> coalesced) + smem tree reduce
//                                                                                 [matvec_q8_dp4a_blk]
//   scalar arm: cast each int8 lane to int32, 4 IMADs  (what Dot::format emits)
//   dp4a   arm: __dp4a on the 4 packed int8 as one int (what the "named lever" would emit)
// The Vector<i8,4> layout is already dp4a-ready: the 4 contiguous int8 of a group reinterpret to one
// int operand with no repack -- so the arms differ in ALU only, never in loads.
//
// COLD: 8192x8192 int8 weights = 64 MB >> GB10 L2, so every iteration re-streams from DRAM (a matvec
// touches each weight once, LRU-evicting earlier reads before the next iter) -- warm 4096^2 (16 MB) is
// L2-resident and lies (~800 GB/s). Timed with CUDA events; scale-relative bit-exact gate vs a CPU ref.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

// ---- the two inner-dot idioms; identical loads, ALU only differs -------------------------------
__device__ __forceinline__ int dot_scalar(int wpacked, int xpacked) {
    // sign-extend each int8 lane to int32 then 4 IMADs -- what cubecl-cpp Dot::format emits after
    // Vector<i32,4>::cast_from widens the packed int8.
    signed char* w = (signed char*)&wpacked;
    signed char* x = (signed char*)&xpacked;
    return (int)w[0]*(int)x[0] + (int)w[1]*(int)x[1] + (int)w[2]*(int)x[2] + (int)w[3]*(int)x[3];
}
__device__ __forceinline__ int dot_dp4a(int wpacked, int xpacked) {
    return __dp4a(wpacked, xpacked, 0); // the named lever: one hardware 4-way int8 dot
}

// ---- i8pack: 1 thread / row (the uncoalesced DSL shape) ----------------------------------------
template<bool USE_DP4A>
__global__ void mv_i8pack(const int* __restrict wq, const int* __restrict xq,
                          const float* __restrict wd, float* __restrict out, int rows, int k) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= rows) return;
    int ng = k/4, nb = k/32, wbase = row*ng, dbase = row*nb;
    float acc = 0.f;
    for (int g=0; g<ng; ++g) {
        int dp = USE_DP4A ? dot_dp4a(wq[wbase+g], xq[g]) : dot_scalar(wq[wbase+g], xq[g]);
        acc += wd[dbase + g/8] * (float)dp;
    }
    out[row] = acc;
}

// ---- blk: 1 block / row, nt threads (the coalesced DSL shape) + smem tree reduce ---------------
template<bool USE_DP4A>
__global__ void mv_blk(const int* __restrict wq, const int* __restrict xq,
                       const float* __restrict wd, float* __restrict out, int rows, int k) {
    extern __shared__ float smem[];
    int row = blockIdx.x, t = threadIdx.x, nt = blockDim.x;
    int ng = k/4, nb = k/32, wbase = row*ng, dbase = row*nb;
    int per = ng/nt;
    float partial = 0.f;
    for (int j=0; j<per; ++j) {
        int g = j*nt + t;
        int dp = USE_DP4A ? dot_dp4a(wq[wbase+g], xq[g]) : dot_scalar(wq[wbase+g], xq[g]);
        partial += wd[dbase + g/8] * (float)dp;
    }
    smem[t] = partial; __syncthreads();
    for (int s=nt/2; s>0; s>>=1) { if (t<s) smem[t]+=smem[t+s]; __syncthreads(); }
    if (t==0) out[row] = smem[0];
}

// ---- ceiling probe: block/row + ILP (vector int4 load = 4 groups/step) + dp4a ------------------
// Not a DSL kernel; measures the coalesced-decode memory roofline this shape can reach so we can say
// how close the DSL blk schedule is to the ceiling (is there any gap left for "hand" to exploit?).
__global__ void mv_blk_ilp(const int* __restrict wq, const int* __restrict xq,
                           const float* __restrict wd, float* __restrict out, int rows, int k) {
    extern __shared__ float smem[];
    int row = blockIdx.x, t = threadIdx.x, nt = blockDim.x;
    int ng = k/4, nb = k/32, wbase = row*ng, dbase = row*nb;
    int per4 = (ng/nt)/4;  // groups/thread in units of 4 (int4 vector load)
    float partial = 0.f;
    const int4* wq4 = (const int4*)wq; const int4* xq4 = (const int4*)xq;
    for (int j=0; j<per4; ++j) {
        int gi = (j*nt + t);              // index in int4 units
        int4 w = wq4[wbase/4 + gi];
        int4 x = xq4[gi];
        int g0 = gi*4;
        partial += wd[dbase + (g0  )/8]*(float)__dp4a(w.x,x.x,0);
        partial += wd[dbase + (g0+1)/8]*(float)__dp4a(w.y,x.y,0);
        partial += wd[dbase + (g0+2)/8]*(float)__dp4a(w.z,x.z,0);
        partial += wd[dbase + (g0+3)/8]*(float)__dp4a(w.w,x.w,0);
    }
    smem[t] = partial; __syncthreads();
    for (int s=nt/2; s>0; s>>=1) { if (t<s) smem[t]+=smem[t+s]; __syncthreads(); }
    if (t==0) out[row] = smem[0];
}

// deterministic data matching check_dp4a's xorshift (int8 weights/act, f32 per-32-block scales)
static void gen(std::vector<int8_t>& wq, std::vector<int8_t>& xq, std::vector<float>& wd, int rows, int k) {
    uint64_t s = 0x9E3779B97F4A7C15ULL;
    auto nxt=[&](){ s^=s<<13; s^=s>>7; s^=s<<17; return s; };
    wq.resize((size_t)rows*k); for(auto&v:wq) v=(int8_t)(nxt()%255);
    xq.resize(k);              for(auto&v:xq) v=(int8_t)(nxt()%255);
    wd.resize((size_t)rows*(k/32)); for(auto&v:wd) v=(float)(nxt()%1000)/8000.f+0.01f;
}
static std::vector<float> cpu_ref(const std::vector<int8_t>&wq,const std::vector<int8_t>&xq,
                                  const std::vector<float>&wd,int rows,int k){
    int nb=k/32; std::vector<float> out(rows);
    for(int r=0;r<rows;++r){ float acc=0; for(int g=0;g<k/4;++g){ int dp=0;
        for(int l=0;l<4;++l) dp += (int)wq[(size_t)r*k+g*4+l]*(int)xq[g*4+l];
        acc += wd[(size_t)r*nb + g/8]*(float)dp; } out[r]=acc; }
    return out;
}
static float scalerel(const std::vector<float>&a,const float*b){
    float sc=1e-20f; for(float v:a) sc=fmaxf(sc,fabsf(v));
    float m=0; for(size_t i=0;i<a.size();++i) m=fmaxf(m,fabsf(a[i]-b[i])); return m/sc;
}

enum Kind { I8_S, I8_D, BLK_S, BLK_D, ILP };
static float run(Kind kind, int nt, const int* wq, const int* xq, const float* wd, float* out,
                 int rows, int k, int iters, std::vector<float>& hostout) {
    dim3 grid, block; size_t shmem=0;
    if (kind==I8_S||kind==I8_D){ block=dim3(64); grid=dim3((rows+63)/64); }
    else { block=dim3(nt); grid=dim3(rows); shmem=nt*sizeof(float); }
    auto launch=[&](){
        switch(kind){
          case I8_S: mv_i8pack<false><<<grid,block>>>(wq,xq,wd,out,rows,k); break;
          case I8_D: mv_i8pack<true ><<<grid,block>>>(wq,xq,wd,out,rows,k); break;
          case BLK_S: mv_blk<false><<<grid,block,shmem>>>(wq,xq,wd,out,rows,k); break;
          case BLK_D: mv_blk<true ><<<grid,block,shmem>>>(wq,xq,wd,out,rows,k); break;
          case ILP:  mv_blk_ilp<<<grid,block,shmem>>>(wq,xq,wd,out,rows,k); break;
        }
    };
    for(int i=0;i<5;++i) launch(); CK(cudaDeviceSynchronize()); // warmup (still cold: 64MB>>L2)
    hostout.resize(rows);
    CK(cudaMemcpy(hostout.data(), out, rows*sizeof(float), cudaMemcpyDeviceToHost));
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a));
    for(int i=0;i<iters;++i) launch();
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms=0; CK(cudaEventElapsedTime(&ms,a,b)); return ms/iters;
}

int main(int argc, char** argv){
    int rows = argc>1?atoi(argv[1]):8192;
    int k    = argc>2?atoi(argv[2]):8192;
    int iters= argc>3?atoi(argv[3]):200;
    int dev=0; cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,dev));
    printf("GPU %s  sm_%d%d  L2=%d MB  (GB10 spec ~273 GB/s LPDDR5X)\n",
           p.name, p.major, p.minor, p.l2CacheSize>>20);
    double footprint_mb = (double)rows*k/1e6;
    printf("shape rows=%d k=%d  weight footprint=%.0f MB (%s L2)  iters=%d\n\n",
           rows,k,footprint_mb, footprint_mb>(p.l2CacheSize>>20)?"COLD >>":"WARM <=", iters);

    std::vector<int8_t> wq,xq; std::vector<float> wd; gen(wq,xq,wd,rows,k);
    std::vector<float> ref = cpu_ref(wq,xq,wd,rows,k);

    int8_t *dwq,*dxq; float *dwd,*dout;
    CK(cudaMalloc(&dwq,wq.size())); CK(cudaMalloc(&dxq,xq.size()));
    CK(cudaMalloc(&dwd,wd.size()*4)); CK(cudaMalloc(&dout,rows*4));
    CK(cudaMemcpy(dwq,wq.data(),wq.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dxq,xq.data(),xq.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dwd,wd.data(),wd.size()*4,cudaMemcpyHostToDevice));

    const int* wqi=(const int*)dwq; const int* xqi=(const int*)dxq;
    double bytes = (double)rows*k; // int8 weights, 1 byte each -- the hand-tuned footprint accounting
    struct Row{const char* tag; Kind kind; int nt;};
    Row rowsv[] = {
        {"i8pack scalar", I8_S, 0}, {"i8pack dp4a  ", I8_D, 0},
        {"blk64  scalar", BLK_S,64}, {"blk64  dp4a  ", BLK_D,64},
        {"blk128 scalar", BLK_S,128},{"blk128 dp4a  ", BLK_D,128},
        {"blk256 scalar", BLK_S,256},{"blk256 dp4a  ", BLK_D,256},
        {"blk256 ilp4dp4a",ILP, 256},
    };
    printf("%-16s  %9s  %8s  %10s  %s\n","kernel","ms","GB/s","GFLOP/s","gate");
    for(auto&rw:rowsv){
        std::vector<float> hostout;
        float ms=run(rw.kind,rw.nt,wqi,xqi,dwd,dout,rows,k,iters,hostout);
        float sr=scalerel(ref,hostout.data());
        printf("%-16s  %9.3f  %8.0f  %10.0f  scale_rel=%.2e %s\n",
               rw.tag, ms, bytes/(ms*1e6), 2.0*rows*k/(ms*1e6), sr, sr<2e-2?"OK":"MISMATCH");
    }
    return 0;
}
