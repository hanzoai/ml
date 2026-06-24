// Production quant.hip kernels + a GPU dequant-to-f16 for the fallback A/B. The dequant kernels mirror
// the canonical GGML to_float (one thread per 256-elem super-block) so the rocBLAS fallback sees the
// same weights the native path consumes -- a fair best-case fallback (the shipped fallback dequants on
// CPU, which is strictly slower).
#include "../src/kernels/quant.hip"

#if defined(__HIPCC__)
__device__ __forceinline__ float bh2f(const uint8_t* p){ __half h; __builtin_memcpy(&h,p,2); return __half2float(h); }

// Q4_K dequant: [n, k] weight -> f16 [n,k] row-major. One thread per super-block (n*k/256 threads).
extern "C" __global__ void deq_q4k_f16(int n,int k,const uint8_t* __restrict__ wq,__half* __restrict__ out){
    int nsb=n*(k/256); int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=nsb) return;
    int row=t/(k/256), sb=t%(k/256);
    const uint8_t* b=wq+((size_t)row*(k/256)+sb)*144;
    float d=bh2f(b), dmin=bh2f(b+2); const uint8_t* sc=b+4; const uint8_t* qs=b+16;
    __half* o=out+(size_t)row*k+sb*256;
    for(int g=0;g<8;g++){ int s,mm;
        if(g<4){ s=sc[g]&63; mm=sc[g+4]&63; }
        else { s=(sc[g+4]&0xF)|((sc[g-4]>>6)<<4); mm=(sc[g+4]>>4)|((sc[g]>>6)<<4); }
        float dl=d*(float)s, ml=dmin*(float)mm; int c=g/2, sh=(g&1)?4:0;
        for(int kk=0;kk<32;kk++){ int q=(qs[c*32+kk]>>sh)&0xF; o[g*32+kk]=__float2half(dl*(float)q-ml); }
    }
}

// Q6_K dequant: canonical layout, per-16 signed scales.
extern "C" __global__ void deq_q6k_f16(int n,int k,const uint8_t* __restrict__ wq,__half* __restrict__ out){
    int nsb=n*(k/256); int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=nsb) return;
    int row=t/(k/256), sb=t%(k/256);
    const uint8_t* b=wq+((size_t)row*(k/256)+sb)*210;
    const uint8_t* ql=b; const uint8_t* qh=b+128; const int8_t* sc=(const int8_t*)(b+192);
    float dd=bh2f(b+208); __half* o=out+(size_t)row*k+sb*256;
    for(int g=0;g<8;g++){ int hf=g/4, quad=g&3; const uint8_t* qll=ql+64*hf; const uint8_t* qhh=qh+32*hf; int base=g*32;
        for(int kk=0;kk<32;kk++){ int q;
            if(quad==0)      q=((qll[kk]    &0xF)|((qhh[kk]&3)<<4))-32;
            else if(quad==1) q=((qll[kk+32] &0xF)|(((qhh[kk]>>2)&3)<<4))-32;
            else if(quad==2) q=((qll[kk]    >>4 )|(((qhh[kk]>>4)&3)<<4))-32;
            else             q=((qll[kk+32] >>4 )|(((qhh[kk]>>6)&3)<<4))-32;
            int pos=base+kk; o[pos]=__float2half(dd*(float)sc[pos/16]*(float)q);
        }
    }
}
#endif
