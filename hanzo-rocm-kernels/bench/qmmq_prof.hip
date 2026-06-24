// Profiling driver: launches ONE qmmq kernel (chosen by argv) many times at a fixed shape so
// rocprofv3 gets stable per-dispatch counters localizing the LDS/VALU/MMA stall. Loads qmmq_kernels.hsaco.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cstring>
#define HC(c) do{hipError_t e=(c); if(e!=hipSuccess){printf("HIP %s %d\n",hipGetErrorString(e),__LINE__);std::exit(1);}}while(0)
static uint16_t f2h(float f){__half x=__float2half(f);uint16_t o;memcpy(&o,&x,2);return o;}
static uint8_t rb(size_t s){return (uint8_t)((s*2654435761ull+(s>>7))&0xff);}
static void pd(uint8_t*b,int o,float d){uint16_t h=f2h(d);b[o]=h&0xff;b[o+1]=h>>8;}
int main(int argc,char**argv){
    const char* which = argc>1?argv[1]:"q8_0";
    int m=512,k=5120,n=8192,iters=argc>2?atoi(argv[2]):60;
    const char* obj = argc>3?argv[3]:"qmmq_kernels.hsaco";
    hipModule_t M; HC(hipModuleLoad(&M,obj));
    hipFunction_t q81,gf;
    HC(hipModuleGetFunction(&q81,M,"quantize_q8_1"));
    char kn[32]; snprintf(kn,sizeof kn,"qmmq_%s_f16",which);
    HC(hipModuleGetFunction(&gf,M,kn));
    int nblk32=k/32, ncol=(n+127)/128, nrow=(m+127)/128;
    int tsz = !strcmp(which,"q8_0")?34 : !strcmp(which,"q4k")?144 : !strcmp(which,"q6k")?210 : 34;
    int blk = !strcmp(which,"q8_0")?32:256;
    std::vector<uint16_t> xh((size_t)m*k,f2h(0.4f));
    int nb=k/blk; std::vector<uint8_t> w((size_t)n*nb*tsz);
    for(int r=0;r<n;r++)for(int b=0;b<nb;b++){uint8_t*p=&w[((size_t)r*nb+b)*tsz];
        for(int i=0;i<tsz;i++)p[i]=rb((size_t)r*7919+b*131+i*17+3);
        if(!strcmp(which,"q8_0"))pd(p,0,0.03f); else if(!strcmp(which,"q4k")){pd(p,0,0.03f);pd(p,2,0.02f);} else pd(p,208,0.004f);}
    __half*d_x,*d_xd,*d_y; int8_t*d_xq; int*d_xs; uint8_t*d_w;
    HC(hipMalloc(&d_x,(size_t)m*k*2));HC(hipMalloc(&d_xq,(size_t)m*k));HC(hipMalloc(&d_xd,(size_t)m*nblk32*2));
    HC(hipMalloc(&d_xs,(size_t)m*nblk32*4));HC(hipMalloc(&d_y,(size_t)m*n*2));HC(hipMalloc(&d_w,w.size()));
    HC(hipMemcpy(d_x,xh.data(),(size_t)m*k*2,hipMemcpyHostToDevice));
    HC(hipMemcpy(d_w,w.data(),w.size(),hipMemcpyHostToDevice));
    int qg=(m*nblk32*32+255)/256; void* qa[]={&m,&k,&d_x,&d_xq,&d_xd,&d_xs};
    HC(hipModuleLaunchKernel(q81,qg,1,1,256,1,1,0,0,qa,nullptr)); HC(hipDeviceSynchronize());
    void* ga[]={&m,&n,&k,&ncol,&d_xq,&d_xd,&d_w,&d_y,&d_xs};
    for(int i=0;i<iters;i++) HC(hipModuleLaunchKernel(gf,(unsigned)(ncol*nrow),1,1,512,1,1,0,0,ga,nullptr));
    HC(hipDeviceSynchronize());
    printf("done %s %s iters=%d shape=%dx%dx%d\n",which,obj,iters,m,k,n);
    return 0;
}
