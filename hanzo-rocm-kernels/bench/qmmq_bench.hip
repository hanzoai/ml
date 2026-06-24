// Microbench: native int8-WMMA prefill GEMM (qmmq_core<WTYPE>) vs the dequant-to-f16 + rocBLAS hgemm
// fallback, on production FFN shapes. Native = quantize_q8/q8_1 + qmmq_<t>_f16 (the full prefill cost).
// Fallback = GPU dequant weight to f16 (best-case; shipped fallback dequants on CPU = far slower) +
// rocBLAS hgemm. Reports us/call, GB/s on the quantized weight read, and speedup.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>

#define HC(c) do{hipError_t e=(c); if(e!=hipSuccess){printf("HIP %s %d\n",hipGetErrorString(e),__LINE__);std::exit(1);}}while(0)
#define RB(c) do{rocblas_status s=(c); if(s!=rocblas_status_success){printf("rocBLAS %d %d\n",(int)s,__LINE__);std::exit(1);}}while(0)

static hipModule_t M; static hipFunction_t F_q8,F_q81,F_qmmq_q4k,F_qmmq_q6k,F_qmmq_q8,F_deq_q4k,F_deq_q6k;
static rocblas_handle RBH;

static uint16_t f2h(float f){__half x=__float2half(f);uint16_t o;memcpy(&o,&x,2);return o;}
static uint8_t rbyte(size_t s){return (uint8_t)((s*2654435761ull+(s>>7))&0xff);}
static void put_d(uint8_t*b,int off,float d){uint16_t h=f2h(d);b[off]=h&0xff;b[off+1]=h>>8;}

static std::vector<uint8_t> build(int n,int k,int blk,int tsz,bool q4k){
    int nblk=k/blk; std::vector<uint8_t> out((size_t)n*nblk*tsz);
    for(int r=0;r<n;r++)for(int b=0;b<nblk;b++){uint8_t*p=&out[((size_t)r*nblk+b)*tsz];
        for(int i=0;i<tsz;i++)p[i]=rbyte((size_t)r*7919+(size_t)b*131+(size_t)i*17+3);
        if(q4k){put_d(p,0,0.03f);put_d(p,2,0.02f);}else{put_d(p,208,0.004f);}
    }
    return out;
}

static double tmin(hipFunction_t f,unsigned gx,unsigned bx,void**args,int it){
    for(int i=0;i<5;i++)HC(hipModuleLaunchKernel(f,gx,1,1,bx,1,1,0,0,args,nullptr));
    HC(hipDeviceSynchronize());
    double best=1e30;
    for(int r=0;r<it;r++){auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<10;i++)HC(hipModuleLaunchKernel(f,gx,1,1,bx,1,1,0,0,args,nullptr));
        HC(hipDeviceSynchronize());auto t1=std::chrono::high_resolution_clock::now();
        best=std::min(best,std::chrono::duration<double,std::micro>(t1-t0).count()/10.0);}
    return best;
}

static void run(const char* name,int m,int k,int n,bool q4k){
    int blk=256,tsz=q4k?144:210; int nblk32=k/32;
    auto wb=build(n,k,blk,tsz,q4k);
    std::vector<uint16_t> xh((size_t)m*k);
    for(size_t i=0;i<(size_t)m*k;i++)xh[i]=f2h(((float)((i*2654435761ull>>8)&0xffff)/65535.0f-0.5f)*2.0f);

    __half *d_x,*d_xd,*d_y,*d_wf16; int8_t*d_xq; int*d_xs; uint8_t*d_wq;
    HC(hipMalloc(&d_x,(size_t)m*k*2)); HC(hipMalloc(&d_xq,(size_t)m*k)); HC(hipMalloc(&d_xd,(size_t)m*nblk32*2));
    HC(hipMalloc(&d_xs,(size_t)m*nblk32*4)); HC(hipMalloc(&d_wq,wb.size())); HC(hipMalloc(&d_y,(size_t)m*n*2));
    HC(hipMalloc(&d_wf16,(size_t)n*k*2));
    HC(hipMemcpy(d_x,xh.data(),(size_t)m*k*2,hipMemcpyHostToDevice));
    HC(hipMemcpy(d_wq,wb.data(),wb.size(),hipMemcpyHostToDevice));

    int total=m*nblk32, qb=256, qg=(total*32+qb-1)/qb, ncol=(n+127)/128, nrow=(m+127)/128;
    hipFunction_t qf=q4k?F_q81:F_q8, gf=q4k?F_qmmq_q4k:F_qmmq_q6k;
    void* qa[]={&m,&k,&d_x,&d_xq,&d_xd,&d_xs};
    void* ga[]={&m,&n,&k,&ncol,&d_xq,&d_xd,&d_wq,&d_y,&d_xs};
    for(int i=0;i<5;i++){HC(hipModuleLaunchKernel(qf,qg,1,1,qb,1,1,0,0,qa,nullptr));
        HC(hipModuleLaunchKernel(gf,(unsigned)(ncol*nrow),1,1,512,1,1,0,0,ga,nullptr));}
    HC(hipDeviceSynchronize());
    double native=1e30;
    for(int r=0;r<20;r++){auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<10;i++){HC(hipModuleLaunchKernel(qf,qg,1,1,qb,1,1,0,0,qa,nullptr));
            HC(hipModuleLaunchKernel(gf,(unsigned)(ncol*nrow),1,1,512,1,1,0,0,ga,nullptr));}
        HC(hipDeviceSynchronize());auto t1=std::chrono::high_resolution_clock::now();
        native=std::min(native,std::chrono::duration<double,std::micro>(t1-t0).count()/10.0);}

    hipFunction_t df=q4k?F_deq_q4k:F_deq_q6k;
    int nsblk=n*(k/256); int dgb=256; unsigned dgg=(unsigned)((nsblk+dgb-1)/dgb);
    void* da[]={&n,&k,&d_wq,&d_wf16};
    double dq=tmin(df,dgg,dgb,da,20);

    __half alpha=__float2half(1.f), beta=__float2half(0.f);
    // Row-major Y[m,n]=X[m,k]*W[n,k]^T  <=>  col-major C(n,m)=opA(W)*opB(X), A=W(n,k) row-major =
    // col-major(k,n) so opA=T; B=X(m,k) row-major = col-major(k,m) so opB=N. lda=k ldb=k ldc=n.
    auto hgemm=[&](){ RB(rocblas_hgemm(RBH,rocblas_operation_transpose,rocblas_operation_none,
        n,m,k,(const rocblas_half*)&alpha,(const rocblas_half*)d_wf16,k,(const rocblas_half*)d_x,k,
        (const rocblas_half*)&beta,(rocblas_half*)d_y,n)); };
    for(int i=0;i<5;i++){HC(hipModuleLaunchKernel(df,dgg,1,1,dgb,1,1,0,0,da,nullptr));hgemm();}
    HC(hipDeviceSynchronize());
    double fb=1e30, gemm_only=1e30;
    for(int r=0;r<20;r++){auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<10;i++){HC(hipModuleLaunchKernel(df,dgg,1,1,dgb,1,1,0,0,da,nullptr));hgemm();}
        HC(hipDeviceSynchronize());auto t1=std::chrono::high_resolution_clock::now();
        fb=std::min(fb,std::chrono::duration<double,std::micro>(t1-t0).count()/10.0);}
    for(int r=0;r<20;r++){auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<10;i++)hgemm();
        HC(hipDeviceSynchronize());auto t1=std::chrono::high_resolution_clock::now();
        gemm_only=std::min(gemm_only,std::chrono::duration<double,std::micro>(t1-t0).count()/10.0);}

    // GEMM-only (exclude the quantize launch) to isolate the qmmq compute from activation-quant cost.
    double gemm_native=1e30;
    for(int r=0;r<20;r++){auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<10;i++)HC(hipModuleLaunchKernel(gf,(unsigned)(ncol*nrow),1,1,512,1,1,0,0,ga,nullptr));
        HC(hipDeviceSynchronize());auto t1=std::chrono::high_resolution_clock::now();
        gemm_native=std::min(gemm_native,std::chrono::duration<double,std::micro>(t1-t0).count()/10.0);}
    double flop=2.0*(double)m*n*k;
    double tflops_n=flop/(gemm_native*1e-6)/1e12, tflops_f=flop/(gemm_only*1e-6)/1e12;
    printf("%-18s m=%d k=%d n=%d | native=%6.1fus (qmmq=%6.1f %.2f TF/s)  fallback=%7.1fus (dq=%6.1f hgemm=%6.1f %.2f TF/s)  speedup=%.2fx  qmmq-vs-hgemm=%.2fx\n",
        name,m,k,n,native,gemm_native,tflops_n,fb,dq,gemm_only,tflops_f,fb/native,gemm_only/gemm_native);
    hipFree(d_x);hipFree(d_xq);hipFree(d_xd);hipFree(d_xs);hipFree(d_wq);hipFree(d_y);hipFree(d_wf16);
}

int main(){
    HC(hipModuleLoad(&M,"qmmq_kernels.hsaco"));
    HC(hipModuleGetFunction(&F_q8,M,"quantize_q8"));
    HC(hipModuleGetFunction(&F_q81,M,"quantize_q8_1"));
    HC(hipModuleGetFunction(&F_qmmq_q4k,M,"qmmq_q4k_f16"));
    HC(hipModuleGetFunction(&F_qmmq_q6k,M,"qmmq_q6k_f16"));
    HC(hipModuleGetFunction(&F_qmmq_q8,M,"qmmq_q8_0_f16"));
    HC(hipModuleGetFunction(&F_deq_q4k,M,"deq_q4k_f16"));
    HC(hipModuleGetFunction(&F_deq_q6k,M,"deq_q6k_f16"));
    RB(rocblas_create_handle(&RBH));

    printf("=== Q4_K FFN ===\n");
    run("Q4K gate/up 14B", 512,5120,17408,true);
    run("Q4K down 14B",    512,17408,5120,true);
    run("Q4K gate/up A3B", 512,2048,6144, true);
    run("Q4K down A3B",    512,6144,2048, true);
    printf("=== Q6_K FFN ===\n");
    run("Q6K down 14B",    512,17408,5120,false);
    run("Q6K gate/up A3B", 512,2048,6144, false);

    // GEMM-only decode-cost probe: same shape, different weight decode complexity. If qmmq is decode-
    // bound, Q8_0 (trivial memcpy decode) >> Q4_K (nibble unpack) >> Q6_K (6-bit reassemble).
    printf("=== GEMM-only weight-decode probe (m=512 k=5120 n=8192, qmmq vs hgemm) ===\n");
    {
        int m=512,k=5120,n=8192,nblk32=k/32,ncol=(n+127)/128,nrow=(m+127)/128;
        std::vector<uint16_t> xh((size_t)m*k,f2h(0.5f));
        auto w8=build(n,k,32,34,true); auto w4=build(n,k,256,144,true); auto w6=build(n,k,256,210,false);
        __half*d_x,*d_xd; int8_t*d_xq; int*d_xs; __half*d_y; uint8_t*d_w8,*d_w4,*d_w6;
        HC(hipMalloc(&d_x,(size_t)m*k*2));HC(hipMalloc(&d_xq,(size_t)m*k));HC(hipMalloc(&d_xd,(size_t)m*nblk32*2));
        HC(hipMalloc(&d_xs,(size_t)m*nblk32*4));HC(hipMalloc(&d_y,(size_t)m*n*2));
        HC(hipMalloc(&d_w8,w8.size()));HC(hipMalloc(&d_w4,w4.size()));HC(hipMalloc(&d_w6,w6.size()));
        HC(hipMemcpy(d_x,xh.data(),(size_t)m*k*2,hipMemcpyHostToDevice));
        HC(hipMemcpy(d_w8,w8.data(),w8.size(),hipMemcpyHostToDevice));
        HC(hipMemcpy(d_w4,w4.data(),w4.size(),hipMemcpyHostToDevice));
        HC(hipMemcpy(d_w6,w6.data(),w6.size(),hipMemcpyHostToDevice));
        int qg=(m*nblk32*32+255)/256;
        void* qa[]={&m,&k,&d_x,&d_xq,&d_xd,&d_xs};
        HC(hipModuleLaunchKernel(F_q81,qg,1,1,256,1,1,0,0,qa,nullptr)); HC(hipDeviceSynchronize());
        double flop=2.0*m*n*k;
        auto probe=[&](const char* nm,hipFunction_t f,uint8_t* w){ void* ga[]={&m,&n,&k,&ncol,&d_xq,&d_xd,&w,&d_y,&d_xs};
            double best=tmin(f,(unsigned)(ncol*nrow),512,ga,30);
            printf("  %-6s qmmq=%7.1fus  %.2f TF/s\n",nm,best,flop/(best*1e-6)/1e12); };
        probe("Q8_0",F_qmmq_q8,d_w8); probe("Q4_K",F_qmmq_q4k,d_w4); probe("Q6_K",F_qmmq_q6k,d_w6);
    }
    return 0;
}
