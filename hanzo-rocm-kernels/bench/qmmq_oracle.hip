// Standalone numeric oracle for the unified native ROCm prefill GEMM (qmmq_core<WTYPE>) on gfx1151.
// Includes the production quant.hip verbatim (so the EXACT kernels under test run), builds Q4_K/Q6_K
// weight tensors in proper GGML layout on host, runs quantize_q8/quantize_q8_1 + qmmq_<t>_f16 on the
// GPU, and checks against a CPU dequant->f32 GEMM reference on the SAME per-32-block int8-requantized
// activation the kernel builds. Gate: nbad==0 within 1% tolerance. Mirrors qmmq_unified_numeric.rs.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>

// quant_test.hip is NOT included here; the kernels live in a separately-compiled code object
// (qmmq_kernels.hsaco), loaded at runtime via the
// HIP module API -- exactly how the Rust rocm_backend loads them. This sidesteps the host-stub gen for
// the arch-gated WMMA kernels and tests the real shipped binary path.
static hipModule_t g_mod;
static hipFunction_t g_q8, g_q4k, g_q6k, g_qq8, g_qq81;

#define HC(call) do { hipError_t e=(call); if(e!=hipSuccess){ \
  printf("HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); std::exit(1);} } while(0)

static float h2f(uint16_t h){ __half x; memcpy(&x,&h,2); return __half2float(x); }
static uint16_t f2h(float f){ __half x=__float2half(f); uint16_t o; memcpy(&o,&x,2); return o; }

// Deterministic generators matching the Rust oracle.
static float val(size_t i){ float x=((float)((i*2654435761ull>>8)&0xffff))/65535.0f; return (x-0.5f)*6.0f; }
static uint8_t rbyte(size_t s){ return (uint8_t)((s*2654435761ull + (s>>7)) & 0xff); }

static void put_d(uint8_t* blk,int off,int r,int b,float lo,float step,int m){
    float d=lo+step*(float)((r+b)%m); uint16_t h=f2h(d); blk[off]=h&0xff; blk[off+1]=h>>8;
}

// Build raw GGML bytes [n rows x k cols] of block type, body pseudo-random, scales forced exact-f16.
template<typename Fill>
static std::vector<uint8_t> build_bytes(int n,int k,int blk,int tsz,Fill fill){
    int nblk=k/blk; std::vector<uint8_t> out((size_t)n*nblk*tsz,0);
    for(int r=0;r<n;r++) for(int b=0;b<nblk;b++){
        uint8_t* p=&out[((size_t)r*nblk+b)*tsz];
        for(int i=0;i<tsz;i++) p[i]=rbyte((size_t)r*7919+(size_t)b*131+(size_t)i*17+3);
        fill(p,r,b);
    }
    return out;
}

// q4k_scale_min on host (identical to the device version in quant.hip).
static void cpu_q4k_sm(const uint8_t* s,int j,int* sc,int* m){
    if(j<4){ *sc=s[j]&63; *m=s[j+4]&63; }
    else { *sc=(s[j+4]&0xF)|((s[j-4]>>6)<<4); *m=(s[j+4]>>4)|((s[j]>>6)<<4); }
}

// to_float Q4_K: d,dmin (f16), 12 scale bytes, 128 qs. 8 sub-blocks of 32; chunk c=g/2 byte c*32+k.
static void deq_q4k(const uint8_t* b,float* o){
    float d=h2f(*(const uint16_t*)b), dmin=h2f(*(const uint16_t*)(b+2));
    const uint8_t* sc=b+4; const uint8_t* qs=b+16;
    for(int g=0;g<8;g++){ int s,m; cpu_q4k_sm(sc,g,&s,&m);
        float dl=d*(float)s, ml=dmin*(float)m; int c=g/2; int sh=(g&1)?4:0;
        for(int k=0;k<32;k++){ int q=(qs[c*32+k]>>sh)&0xF; o[g*32+k]=dl*(float)q-ml; }
    }
}

// to_float Q6_K: ql[128], qh[64], scales[16] (signed i8), d (f16). Canonical llama layout: per 128-
// half, decoded element positions 0..127 are (low q4[l], low q4[l+32], high q4[l], high q4[l+32]);
// scale for decoded position p is scales[p/16] (SCALE_STEP=16, one signed scale per 16 elements).
static void deq_q6k(const uint8_t* b,float* o){
    const uint8_t* ql=b; const uint8_t* qh=b+128; const int8_t* sc=(const int8_t*)(b+192);
    float d=h2f(*(const uint16_t*)(b+208));
    for(int g=0;g<8;g++){ int hf=g/4, quad=g&3; const uint8_t* qll=ql+64*hf; const uint8_t* qhh=qh+32*hf;
        int base=g*32; // decoded position of this quadrant's first element
        for(int k=0;k<32;k++){ int q;
            if(quad==0)      q=((qll[k]    &0xF)|((qhh[k]&3)<<4))-32;
            else if(quad==1) q=((qll[k+32] &0xF)|(((qhh[k]>>2)&3)<<4))-32;
            else if(quad==2) q=((qll[k]    >>4 )|(((qhh[k]>>4)&3)<<4))-32;
            else             q=((qll[k+32] >>4 )|(((qhh[k]>>6)&3)<<4))-32;
            int pos=base+k; float scl=d*(float)sc[pos/16];
            o[pos]=scl*(float)q;
        }
    }
}

// Per-32-block int8 requantization of one activation row (kernel's quantize_q8 reconstructed to f32).
static void requant_row(const float* x,float* out,int k){
    for(int o=0;o<k;o+=32){ float amax=0; for(int j=0;j<32;j++) amax=std::max(amax,std::fabs(x[o+j]));
        float inv=amax>0?127.0f/amax:0.0f, d=amax/127.0f;
        for(int j=0;j<32;j++){ int q=(int)std::lround(x[o+j]*inv); q=std::max(-127,std::min(127,q)); out[o+j]=(float)q*d; }
    }
}

typedef void(*deqfn)(const uint8_t*,float*);

static int check(const char* name,const char* kern,bool asym,int m,int n,int k,int blk,int tsz,deqfn deq,
                 const std::vector<uint8_t>& wq){
    int nblk32=k/32; int nblk=k/blk;
    // X[m,k] f16
    std::vector<uint16_t> xh((size_t)m*k); std::vector<float> xf((size_t)m*k);
    for(size_t i=0;i<(size_t)m*k;i++){ float v=val(i); xh[i]=f2h(v); xf[i]=h2f(xh[i]); }

    // Device buffers
    __half *d_x; int8_t *d_xq; __half *d_xd; int *d_xs; uint8_t *d_wq; __half *d_y;
    HC(hipMalloc(&d_x,(size_t)m*k*2)); HC(hipMalloc(&d_xq,(size_t)m*k));
    HC(hipMalloc(&d_xd,(size_t)m*nblk32*2)); HC(hipMalloc(&d_xs,(size_t)m*nblk32*4));
    HC(hipMalloc(&d_wq,wq.size())); HC(hipMalloc(&d_y,(size_t)m*n*2));
    HC(hipMemcpy(d_x,xh.data(),(size_t)m*k*2,hipMemcpyHostToDevice));
    HC(hipMemcpy(d_wq,wq.data(),wq.size(),hipMemcpyHostToDevice));
    HC(hipMemset(d_xs,0,(size_t)m*nblk32*4));

    // quantize activations (q8 for sym, q8_1 for asym)
    int total=m*nblk32; int qb=256; int qg=(total*32+qb-1)/qb;
    {
        void* args[]={&m,&k,&d_x,&d_xq,&d_xd,&d_xs};
        hipFunction_t f=asym?g_qq81:g_qq8;
        HC(hipModuleLaunchKernel(f,qg,1,1,qb,1,1,0,0,args,nullptr));
    }
    HC(hipDeviceSynchronize());

    // launch the prefill GEMM via the module API (the real shipped code object).
    int ncol=(n+127)/128, nrow=(m+127)/128;
    unsigned gx=ncol*nrow;
    hipFunction_t gf = !strcmp(kern,"q4k")?g_q4k : !strcmp(kern,"q6k")?g_q6k : g_q8;
    {
        void* args[]={&m,&n,&k,&ncol,&d_xq,&d_xd,&d_wq,&d_y,&d_xs};
        HC(hipModuleLaunchKernel(gf,gx,1,1,512,1,1,0,0,args,nullptr));
    }
    HC(hipDeviceSynchronize());

    std::vector<uint16_t> yh((size_t)m*n);
    HC(hipMemcpy(yh.data(),d_y,(size_t)m*n*2,hipMemcpyDeviceToHost));

    // CPU oracle: per row, requant activation row, dot real-dequant weight rows.
    std::vector<float> deqbuf(blk);
    std::vector<float> xr(k);
    double max_err=0; int nbad=0; float scale=1;
    std::vector<float> refrow(n);
    for(int mm=0;mm<m;mm++){
        requant_row(&xf[(size_t)mm*k],xr.data(),k);
        for(int nn=0;nn<n;nn++){
            double acc=0;
            for(int b=0;b<nblk;b++){
                deq(&wq[((size_t)nn*nblk+b)*tsz],deqbuf.data());
                for(int j=0;j<blk;j++) acc+=(double)deqbuf[j]*(double)xr[b*blk+j];
            }
            float r=(float)acc; refrow[nn]=r; scale=std::max(scale,std::fabs(r));
            float g=h2f(yh[(size_t)mm*n+nn]); float e=std::fabs(g-r);
            if(e>max_err) max_err=e;
        }
        // tolerance computed per full-tensor scale below; recompute pass after we know scale
        (void)refrow;
    }
    // second pass for nbad with final tol
    float tol=0.01f*std::max(scale,1.0f);
    max_err=0; nbad=0;
    for(int mm=0;mm<m;mm++){
        requant_row(&xf[(size_t)mm*k],xr.data(),k);
        for(int nn=0;nn<n;nn++){
            double acc=0;
            for(int b=0;b<nblk;b++){
                deq(&wq[((size_t)nn*nblk+b)*tsz],deqbuf.data());
                for(int j=0;j<blk;j++) acc+=(double)deqbuf[j]*(double)xr[b*blk+j];
            }
            float r=(float)acc; float g=h2f(yh[(size_t)mm*n+nn]); float e=std::fabs(g-r);
            if(e>max_err) max_err=e; if(e>tol) nbad++;
        }
    }
    printf("%-7s m=%d n=%d k=%d nbad=%d/%d max_err=%.5f (tol %.5f) scale=%.3f  %s\n",
           name,m,n,k,nbad,m*n,max_err,tol,scale, nbad==0?"PASS":"FAIL");
    hipFree(d_x);hipFree(d_xq);hipFree(d_xd);hipFree(d_xs);hipFree(d_wq);hipFree(d_y);
    return nbad;
}

int main(){
    HC(hipModuleLoad(&g_mod,"qmmq_kernels.hsaco"));
    HC(hipModuleGetFunction(&g_qq8 ,g_mod,"quantize_q8"));
    HC(hipModuleGetFunction(&g_qq81,g_mod,"quantize_q8_1"));
    HC(hipModuleGetFunction(&g_q8  ,g_mod,"qmmq_q8_0_f16"));
    HC(hipModuleGetFunction(&g_q4k ,g_mod,"qmmq_q4k_f16"));
    HC(hipModuleGetFunction(&g_q6k ,g_mod,"qmmq_q6k_f16"));
    int fails=0;
    int s256[][3]={{200,160,256},{130,130,512},{64,256,1024},{37,320,768},{512,4096,4096}};
    int s32[][3]={{200,160,128},{130,130,256},{256,512,1024},{64,256,4096}};

    printf("=== Q8_0 (sym, re-validate proven path) ===\n");
    for(auto&s:s32){ auto w=build_bytes(s[1],s[2],32,34,[](uint8_t*b,int r,int bb){put_d(b,0,r,bb,0.0625f,0.03125f,5);});
        // Q8_0 deq inline:
        fails+=check("Q8_0","q8_0",false,s[0],s[1],s[2],32,34,
            [](const uint8_t* b,float* o){ float d=h2f(*(const uint16_t*)b); const int8_t* q=(const int8_t*)(b+2); for(int i=0;i<32;i++) o[i]=d*(float)q[i]; }, w);
    }
    printf("=== Q4_K (ASYMMETRIC -- headline) ===\n");
    for(auto&s:s256){ auto w=build_bytes(s[1],s[2],256,144,[](uint8_t*b,int r,int bb){put_d(b,0,r,bb,0.0625f,0.03125f,5);put_d(b,2,r,bb,0.03125f,0.015625f,4);});
        fails+=check("Q4_K","q4k",true,s[0],s[1],s[2],256,144,deq_q4k,w);
    }
    printf("=== Q6_K (sym, per-16 signed scales) ===\n");
    for(auto&s:s256){ auto w=build_bytes(s[1],s[2],256,210,[](uint8_t*b,int r,int bb){put_d(b,208,r,bb,0.0078125f,0.00390625f,5);});
        fails+=check("Q6_K","q6k",false,s[0],s[1],s[2],256,210,deq_q6k,w);
    }
    printf("\n%s total_fails=%d\n", fails==0?"ALL PASS":"SOME FAIL", fails);
    return fails==0?0:1;
}
