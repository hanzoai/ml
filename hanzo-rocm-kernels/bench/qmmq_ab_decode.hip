// Thermally-controlled A/B: old (byte-loop) vs new (SWAR) qmmq decode, interleaved per-iteration so
// both see the same GPU thermal state (the APU throttles under sustained load, so absolute numbers
// drift -- only an interleaved ratio is trustworthy). Reports median us + TF/s for Q4_K and Q6_K.
// Loads k_old.hsaco (pre-SWAR decode_w_half, checked out from git by run.sh) and k_new.hsaco (the
// shipped SWAR decode). Result gfx1151: Q6_K 1.70-1.74x (decode WAS the bottleneck), Q4_K ~1.00x.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cstring>
#include <algorithm>
#define HC(c) do{hipError_t e=(c); if(e!=hipSuccess){printf("HIP %s %d\n",hipGetErrorString(e),__LINE__);std::exit(1);}}while(0)
static uint16_t f2h(float f){__half x=__float2half(f);uint16_t o;memcpy(&o,&x,2);return o;}
static uint8_t rb(size_t s){return (uint8_t)((s*2654435761ull+(s>>7))&0xff);}
static void pd(uint8_t*b,int o,float d){uint16_t h=f2h(d);b[o]=h&0xff;b[o+1]=h>>8;}

int main(int argc,char**argv){
    int m=512,k=5120,n=8192; if(argc>3){m=atoi(argv[1]);k=atoi(argv[2]);n=atoi(argv[3]);}
    hipModule_t Mo,Mn; HC(hipModuleLoad(&Mo,"k_old.hsaco")); HC(hipModuleLoad(&Mn,"k_new.hsaco"));
    hipFunction_t q81o,q81n,q4o,q4n,q6o,q6n;
    HC(hipModuleGetFunction(&q81o,Mo,"quantize_q8_1")); HC(hipModuleGetFunction(&q81n,Mn,"quantize_q8_1"));
    HC(hipModuleGetFunction(&q4o,Mo,"qmmq_q4k_f16"));   HC(hipModuleGetFunction(&q4n,Mn,"qmmq_q4k_f16"));
    HC(hipModuleGetFunction(&q6o,Mo,"qmmq_q6k_f16"));   HC(hipModuleGetFunction(&q6n,Mn,"qmmq_q6k_f16"));
    int nblk32=k/32, ncol=(n+127)/128, nrow=(m+127)/128;
    std::vector<uint16_t> xh((size_t)m*k,f2h(0.4f));
    auto build=[&](int tsz,bool q4){int nb=k/256;std::vector<uint8_t>o((size_t)n*nb*tsz);
        for(int r=0;r<n;r++)for(int b=0;b<nb;b++){uint8_t*p=&o[((size_t)r*nb+b)*tsz];
        for(int i=0;i<tsz;i++)p[i]=rb((size_t)r*7919+b*131+i*17+3); if(q4){pd(p,0,0.03f);pd(p,2,0.02f);}else pd(p,208,0.004f);}return o;};
    auto w4=build(144,true), w6=build(210,false);
    __half*d_x,*d_xd,*d_y; int8_t*d_xq; int*d_xs; uint8_t*d_w4,*d_w6;
    HC(hipMalloc(&d_x,(size_t)m*k*2));HC(hipMalloc(&d_xq,(size_t)m*k));HC(hipMalloc(&d_xd,(size_t)m*nblk32*2));
    HC(hipMalloc(&d_xs,(size_t)m*nblk32*4));HC(hipMalloc(&d_y,(size_t)m*n*2));
    HC(hipMalloc(&d_w4,w4.size()));HC(hipMalloc(&d_w6,w6.size()));
    HC(hipMemcpy(d_x,xh.data(),(size_t)m*k*2,hipMemcpyHostToDevice));
    HC(hipMemcpy(d_w4,w4.data(),w4.size(),hipMemcpyHostToDevice));
    HC(hipMemcpy(d_w6,w6.data(),w6.size(),hipMemcpyHostToDevice));
    int qg=(m*nblk32*32+255)/256; void* qa[]={&m,&k,&d_x,&d_xq,&d_xd,&d_xs};
    HC(hipModuleLaunchKernel(q81n,qg,1,1,256,1,1,0,0,qa,nullptr)); HC(hipDeviceSynchronize());
    double flop=2.0*m*n*k;
    auto t1=[&](hipFunction_t f,uint8_t*w)->double{ void* ga[]={&m,&n,&k,&ncol,&d_xq,&d_xd,&w,&d_y,&d_xs};
        auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<10;i++)HC(hipModuleLaunchKernel(f,(unsigned)(ncol*nrow),1,1,512,1,1,0,0,ga,nullptr));
        HC(hipDeviceSynchronize()); auto te=std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double,std::micro>(te-t0).count()/10.0; };
    // warm both
    for(int i=0;i<8;i++){t1(q4o,d_w4);t1(q4n,d_w4);t1(q6o,d_w6);t1(q6n,d_w6);}
    std::vector<double> o4,n4,o6,n6;
    for(int r=0;r<40;r++){ o4.push_back(t1(q4o,d_w4)); n4.push_back(t1(q4n,d_w4));
                           o6.push_back(t1(q6o,d_w6)); n6.push_back(t1(q6n,d_w6)); }
    auto med=[&](std::vector<double>&v){std::sort(v.begin(),v.end());return v[v.size()/2];};
    double mo4=med(o4),mn4=med(n4),mo6=med(o6),mn6=med(n6);
    printf("A/B interleaved m=%d k=%d n=%d (median of 40)\n",m,k,n);
    printf("  Q4_K  old=%7.1fus (%.2f TF/s)  new=%7.1fus (%.2f TF/s)  speedup=%.3fx\n",
        mo4,flop/(mo4*1e-6)/1e12,mn4,flop/(mn4*1e-6)/1e12,mo4/mn4);
    printf("  Q6_K  old=%7.1fus (%.2f TF/s)  new=%7.1fus (%.2f TF/s)  speedup=%.3fx\n",
        mo6,flop/(mo6*1e-6)/1e12,mn6,flop/(mn6*1e-6)/1e12,mo6/mn6);
    return 0;
}
