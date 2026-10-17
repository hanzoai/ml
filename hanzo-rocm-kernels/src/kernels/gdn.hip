// Fused gated-delta-rule scan (GDN linear-attention recurrence), f32 end-to-end.
// One thread per (batch*head, v_column): a private state column st[kd] in registers, sequential
// over the sequence, no cross-thread communication -- the exact math of hanzo-kernel gdn_scan
// (CPU-gated against the engine's portable scan). q arrives pre-scaled by 1/sqrt(kd).
// Layout: q,k: [bh, seq, kd]  v,out: [bh, seq, vd]  g,beta: [bh, seq]  state: [bh, kd, vd] (in-place).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

extern "C" __global__ void gdn_scan_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    float* __restrict__ state,
    float* __restrict__ out,
    unsigned bh,
    unsigned seq,
    unsigned kd,
    unsigned vd)
{
    const unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= bh * vd) return;
    const unsigned b = col / vd;
    const unsigned vi = col % vd;
    const size_t qk_base = (size_t)b * seq * kd;
    const size_t v_base = (size_t)b * seq * vd;
    const size_t gb_base = (size_t)b * seq;

    // kd is model-fixed (<= 128); the host op bails above that.
    float st[128];
    for (unsigned j = 0; j < kd; ++j) st[j] = state[(size_t)b * kd * vd + (size_t)j * vd + vi];

    for (unsigned t = 0; t < seq; ++t) {
        const float* kt = k + qk_base + (size_t)t * kd;
        const float decay = expf(g[gb_base + t]);
        const float beta_t = beta[gb_base + t];
        const float v_t = v[v_base + (size_t)t * vd + vi];
        float kv = 0.0f;
        for (unsigned j = 0; j < kd; ++j) {
            st[j] *= decay;
            kv += st[j] * kt[j];
        }
        const float delta = (v_t - kv) * beta_t;
        const float* qt = q + qk_base + (size_t)t * kd;
        float y = 0.0f;
        for (unsigned j = 0; j < kd; ++j) {
            st[j] += kt[j] * delta;
            y += st[j] * qt[j];
        }
        out[v_base + (size_t)t * vd + vi] = y;
    }
    for (unsigned j = 0; j < kd; ++j) state[(size_t)b * kd * vd + (size_t)j * vd + vi] = st[j];
}

// Causal depthwise conv1d + silu over a GDN layer's projected q|k|v, read and written in the
// projection's own [batch, seq, ch] layout, carrying the last KW inputs per channel in `state`
// [batch, ch, KW] (f16, bf16 or f32, rewritten in place). With h = history ++ x, the history being
// the state (a continuation) or zeros (`fresh`, a sequence start):
//     out[t] = silu(sum_j w[j] * h[t + 1 + j]),   state <- the last KW entries of h.
// One thread per (batch, channel), sequential over the sequence with the window in registers;
// consecutive threads are consecutive channels, so every load and store is coalesced.
__device__ __forceinline__ float gdn_ld(const float* p) { return *p; }
__device__ __forceinline__ float gdn_ld(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float gdn_ld(const hip_bfloat16* p) { return (float)(*p); }
__device__ __forceinline__ void gdn_st(float* p, float v) { *p = v; }
__device__ __forceinline__ void gdn_st(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void gdn_st(hip_bfloat16* p, float v) { *p = hip_bfloat16(v); }

template <typename ST, int KW>
__device__ __forceinline__ void gdn_conv_core(
    const float* __restrict__ x, const float* __restrict__ w, ST* __restrict__ state,
    float* __restrict__ out, unsigned batch, unsigned seq, unsigned ch, int fresh)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch * ch) return;
    const unsigned b = i / ch, c = i % ch;
    ST* st = state + (size_t)i * KW;
    float h[KW], wr[KW];
    #pragma unroll
    for (int j = 0; j < KW; ++j) {
        h[j] = fresh ? 0.0f : gdn_ld(st + j);
        wr[j] = w[(size_t)c * KW + j];
    }
    const float* xb = x + (size_t)b * seq * ch + c;
    float* ob = out + (size_t)b * seq * ch + c;
    for (unsigned t = 0; t < seq; ++t) {
        #pragma unroll
        for (int j = 0; j + 1 < KW; ++j) h[j] = h[j + 1];
        h[KW - 1] = xb[(size_t)t * ch];
        float acc = 0.0f;
        #pragma unroll
        for (int j = 0; j < KW; ++j) acc += wr[j] * h[j];
        ob[(size_t)t * ch] = acc / (1.0f + expf(-acc));
    }
    #pragma unroll
    for (int j = 0; j < KW; ++j) gdn_st(st + j, h[j]);
}

#define GDN_CONV(SUF, ST)                                                                        \
    extern "C" __global__ void gdn_conv4_##SUF(                                                  \
        const float* __restrict__ x, const float* __restrict__ w, ST* __restrict__ state,        \
        float* __restrict__ out, unsigned batch, unsigned seq, unsigned ch, int fresh) {         \
        gdn_conv_core<ST, 4>(x, w, state, out, batch, seq, ch, fresh);                           \
    }
GDN_CONV(f32, float)
GDN_CONV(f16, __half)
GDN_CONV(bf16, hip_bfloat16)
#undef GDN_CONV

// The whole gated-delta-net mixer after the conv, one block per (batch, v head), a thread per
// value column, the head's state column in registers for the whole sequence:
//   q, k   = l2norm(q), l2norm(k) of k head (h mod hk), the GGUF tiled head order; q * 1/sqrt(D)
//   beta   = sigmoid(b),  decay = exp(A * softplus(a + dt_bias))
//   S      = decay * S;  S += k (beta (v - S^T k))^T;  y = S^T q
//   out    = rmsnorm(y) * norm_w * act(z),  act = sigmoid or silu
// Each wave keeps its own copy of the token's normalized q and k (lanes load 4 each, a wave-wide
// shuffle sums the squares), so the only block-wide sync per token is the one the output norm
// needs across the head's four waves. qkv is the conv output [rows, qkv_stride] = [q | k | v];
// a, b are [rows, hv]; z and out are [rows, hv * DV]; state is [batch, hv, D, DV] f32, in place.
template <int D, int DV>
__device__ __forceinline__ void gdn_mixer_core(
    const float* __restrict__ qkv, unsigned qkv_stride,
    const float* __restrict__ a, const float* __restrict__ bgate, const float* __restrict__ z,
    const float* __restrict__ A, const float* __restrict__ dt_bias,
    const float* __restrict__ norm_w, float* __restrict__ state, float* __restrict__ out,
    unsigned seq, unsigned hk, unsigned hv, float eps, int sigmoid_gate)
{
    static_assert(D == 128 && DV % 32 == 0, "a wave holds a 128-wide key as 32 float4");
    constexpr int NW = DV / 32;
    const unsigned b = blockIdx.x / hv, h = blockIdx.x % hv, kh = h % hk;
    const unsigned col = threadIdx.x, wave = col >> 5, lane = col & 31;
    __shared__ float4 sk[NW][D / 4];
    __shared__ float4 sq[NW][D / 4];
    __shared__ float red[2][NW];

    float* sp = state + ((size_t)(b * hv + h) * D) * DV + col;
    float st[D];
    #pragma unroll
    for (int j = 0; j < D; ++j) st[j] = sp[(size_t)j * DV];

    const float Ah = A[h], dth = dt_bias[h], wcol = norm_w[col];
    const float qscale = rsqrtf((float)D);
    for (unsigned t = 0; t < seq; ++t) {
        const size_t row = (size_t)b * seq + t;
        const float* base = qkv + row * qkv_stride;
        const float4 q4 = reinterpret_cast<const float4*>(base + kh * D)[lane];
        const float4 k4 = reinterpret_cast<const float4*>(base + (hk + kh) * D)[lane];
        const float vt = base[2 * hk * D + h * DV + col];
        float ssq = q4.x * q4.x + q4.y * q4.y + q4.z * q4.z + q4.w * q4.w;
        float ssk = k4.x * k4.x + k4.y * k4.y + k4.z * k4.z + k4.w * k4.w;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            ssq += __shfl_xor(ssq, o, 32);
            ssk += __shfl_xor(ssk, o, 32);
        }
        const float rq = rsqrtf(ssq + 1e-6f), rk = rsqrtf(ssk + 1e-6f);
        sq[wave][lane] = make_float4(q4.x * rq * qscale, q4.y * rq * qscale,
                                     q4.z * rq * qscale, q4.w * rq * qscale);
        sk[wave][lane] = make_float4(k4.x * rk, k4.y * rk, k4.z * rk, k4.w * rk);
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");

        const float beta = 1.0f / (1.0f + expf(-bgate[row * hv + h]));
        const float decay = expf(Ah * logf(1.0f + expf(a[row * hv + h] + dth)));
        float kv = 0.0f;
        #pragma unroll
        for (int j = 0; j < D / 4; ++j) {
            const float4 kk = sk[wave][j];
            st[4 * j + 0] *= decay; kv += st[4 * j + 0] * kk.x;
            st[4 * j + 1] *= decay; kv += st[4 * j + 1] * kk.y;
            st[4 * j + 2] *= decay; kv += st[4 * j + 2] * kk.z;
            st[4 * j + 3] *= decay; kv += st[4 * j + 3] * kk.w;
        }
        const float delta = (vt - kv) * beta;
        float y = 0.0f;
        #pragma unroll
        for (int j = 0; j < D / 4; ++j) {
            const float4 kk = sk[wave][j];
            const float4 qq = sq[wave][j];
            st[4 * j + 0] += kk.x * delta; y += st[4 * j + 0] * qq.x;
            st[4 * j + 1] += kk.y * delta; y += st[4 * j + 1] * qq.y;
            st[4 * j + 2] += kk.z * delta; y += st[4 * j + 2] * qq.z;
            st[4 * j + 3] += kk.w * delta; y += st[4 * j + 3] * qq.w;
        }

        float ss = y * y;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o, 32);
        if (lane == 0) red[t & 1][wave] = ss;
        __syncthreads();
        ss = 0.0f;
        #pragma unroll
        for (int w = 0; w < NW; ++w) ss += red[t & 1][w];
        const float zv = z[row * hv * DV + h * DV + col];
        const float gate = sigmoid_gate ? 1.0f / (1.0f + expf(-zv)) : zv / (1.0f + expf(-zv));
        out[row * hv * DV + h * DV + col] = y * rsqrtf(ss / (float)DV + eps) * wcol * gate;
    }
    #pragma unroll
    for (int j = 0; j < D; ++j) sp[(size_t)j * DV] = st[j];
}

extern "C" __global__ void __launch_bounds__(128) gdn_mixer_d128_f32(
    const float* __restrict__ qkv, unsigned qkv_stride,
    const float* __restrict__ a, const float* __restrict__ bgate, const float* __restrict__ z,
    const float* __restrict__ A, const float* __restrict__ dt_bias,
    const float* __restrict__ norm_w, float* __restrict__ state, float* __restrict__ out,
    unsigned seq, unsigned hk, unsigned hv, float eps, int sigmoid_gate)
{
    gdn_mixer_core<128, 128>(qkv, qkv_stride, a, bgate, z, A, dt_bias, norm_w, state, out,
                             seq, hk, hv, eps, sigmoid_gate);
}
