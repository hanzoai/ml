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

// Rounding at a store point: the value an f32 tensor holds after `to_dtype(rd)` and back.
// rd: 0 keeps f32, 1 rounds to f16, 2 to bf16 (both round-to-nearest-even).
__device__ __forceinline__ float gdn_round(float v, int rd) {
    if (rd == 1) return __half2float(__float2half(v));
    if (rd == 2) return (float)hip_bfloat16(v);
    return v;
}

// Causal depthwise conv1d + silu over a GDN layer's projected q|k|v, read and written in the
// projection's own [batch, seq, ch] layout, carrying the last KW inputs per channel in `state`
// [batch, ch, KW] (f16, bf16 or f32, rewritten in place). With h = history ++ x, the history being
// the state (a continuation) or zeros (`fresh`, a sequence start):
//     out[t] = round(silu(((p0 + p1) + p2) + p3)),   p_j = round(h[t + 1 + j] * w[j]),
// the state <- the last KW entries of h. `round` is `gdn_round(., rd)`: each tap's product and the
// output are stored in the activation dtype, as the engine's composed conv stores them (one
// multiply, one rounding, then an f32 add, tap by tap in order; silu as x / (1 + e^-x)). The
// products and sums are unfused, so rd = 0 is the same arithmetic without the roundings.
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
    float* __restrict__ out, unsigned batch, unsigned seq, unsigned ch, int fresh, int rd)
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
        float acc = gdn_round(__fmul_rn(h[0], wr[0]), rd);
        #pragma unroll
        for (int j = 1; j < KW; ++j) acc = __fadd_rn(acc, gdn_round(__fmul_rn(h[j], wr[j]), rd));
        ob[(size_t)t * ch] = gdn_round(acc / (1.0f + expf(-acc)), rd);
    }
    #pragma unroll
    for (int j = 0; j < KW; ++j) gdn_st(st + j, h[j]);
}

#define GDN_CONV(SUF, ST)                                                                        \
    extern "C" __global__ void gdn_conv4_##SUF(                                                  \
        const float* __restrict__ x, const float* __restrict__ w, ST* __restrict__ state,        \
        float* __restrict__ out, unsigned batch, unsigned seq, unsigned ch, int fresh, int rd) { \
        gdn_conv_core<ST, 4>(x, w, state, out, batch, seq, ch, fresh, rd);                       \
    }
GDN_CONV(f32, float)
GDN_CONV(f16, __half)
GDN_CONV(bf16, hip_bfloat16)
#undef GDN_CONV

// The sum of squares of a 128-wide row as the ROCm `rmsnorm` kernel forms it for a row narrower
// than 1024 (one 32-lane wave): lane L accumulates elements L, L+32, L+64, L+96 in that order,
// then a butterfly over offsets 16, 8, 4, 2, 1. Every lane returns the same total.
__device__ __forceinline__ float gdn_sumsq128(const float* __restrict__ row, unsigned lane) {
    float tmp = 0.0f;
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        const float v = row[lane + 32 * m];
        tmp += v * v;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) tmp += __shfl_xor(tmp, o, 32);
    return tmp;
}

// The whole gated-delta-net mixer after the conv, one block per (batch, v head), a thread per
// value column, the head's state column in registers for the whole sequence. It is the engine's
// composed ROCm path in one launch, the same arithmetic in the same order:
//   q, k   = round(l2norm(q)), round(l2norm(k)) of k head (h mod hk), the GGUF tiled order, where
//            l2norm(x) = (x · rsqrt(Σx²/128 + eps/128)) · (1/√128), the `rms_norm` form of it;
//            then q · 1/√128
//   beta   = 1 / (1 + e^-b),  g = A · softplus(a + dt_bias) (identity past 20), decay = e^g
//   S      = decay · S;  S += k (beta (v - S^T k))^T;  y = round(S^T q)
//   out    = round((y · rsqrt(Σy²/128 + eps)) · norm_w · act(z)),  act = sigmoid or silu
// `round` is `gdn_round(., rd)`. The norms reduce as `gdn_sumsq128` does, reading the row from
// shared memory, so each wave holds the token's q, k and (after the step) y rows in full: the
// only block-wide sync per token is the one y needs across the head's four waves. qkv is the
// conv output [rows, qkv_stride] = [q | k | v]; a, b are [rows, hv]; z and out are
// [rows, hv * DV]; state is [batch, hv, D, DV] f32, in place.
template <int D, int DV>
__device__ __forceinline__ void gdn_mixer_core(
    const float* __restrict__ qkv, unsigned qkv_stride,
    const float* __restrict__ a, const float* __restrict__ bgate, const float* __restrict__ z,
    const float* __restrict__ A, const float* __restrict__ dt_bias,
    const float* __restrict__ norm_w, float* __restrict__ state, float* __restrict__ out,
    unsigned seq, unsigned hk, unsigned hv, float eps, int sigmoid_gate, int rd)
{
    static_assert(D == 128 && DV == 128, "the norms reduce 128-wide rows as one wave");
    constexpr int NW = DV / 32;
    const unsigned b = blockIdx.x / hv, h = blockIdx.x % hv, kh = h % hk;
    const unsigned col = threadIdx.x, wave = col >> 5, lane = col & 31;
    __shared__ float sq[NW][D];
    __shared__ float sk[NW][D];
    __shared__ float sy[2][DV];

    float* sp = state + ((size_t)(b * hv + h) * D) * DV + col;
    float st[D];
    #pragma unroll
    for (int j = 0; j < D; ++j) st[j] = sp[(size_t)j * DV];

    const float Ah = A[h], dth = dt_bias[h], wcol = norm_w[col];
    // The l2 norm's weight and eps as the engine builds them: ones scaled by 1/√128, eps / 128.
    const float unit = (float)0.08838834764831845;
    const float l2eps = (float)(1e-6 / 128.0);
    for (unsigned t = 0; t < seq; ++t) {
        const size_t row = (size_t)b * seq + t;
        const float* base = qkv + row * qkv_stride;
        const float* qrow = base + kh * D;
        const float* krow = base + (hk + kh) * D;
        const float vt = base[2 * hk * D + h * DV + col];
        const float rq = rsqrtf(gdn_sumsq128(qrow, lane) / 128.0f + l2eps);
        const float rk = rsqrtf(gdn_sumsq128(krow, lane) / 128.0f + l2eps);
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const unsigned e = lane + 32 * m;
            sq[wave][e] = gdn_round(rq * qrow[e] * unit, rd) * unit;
            sk[wave][e] = gdn_round(rk * krow[e] * unit, rd);
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");

        const float bt = bgate[row * hv + h];
        const float beta = 1.0f / (expf(-bt) + 1.0f);
        const float xa = a[row * hv + h] + dth;
        const float sp_pos = xa + logf(expf(-xa) + 1.0f);
        const float sp_neg = logf(expf(xa) + 1.0f);
        const float softplus = xa <= 20.0f ? (xa > 0.0f ? sp_pos : sp_neg) : xa;
        const float decay = expf(Ah * softplus);

        // The recurrence exactly as `gdn_scan_f32` steps it.
        const float* kt = sk[wave];
        const float* qt = sq[wave];
        float kv = 0.0f;
        #pragma unroll
        for (int j = 0; j < D; ++j) {
            st[j] *= decay;
            kv += st[j] * kt[j];
        }
        const float delta = (vt - kv) * beta;
        float y = 0.0f;
        #pragma unroll
        for (int j = 0; j < D; ++j) {
            st[j] += kt[j] * delta;
            y += st[j] * qt[j];
        }
        y = gdn_round(y, rd);

        sy[t & 1][col] = y;
        __syncthreads();
        const float ry = rsqrtf(gdn_sumsq128(sy[t & 1], lane) / 128.0f + eps);
        const float zv = z[row * hv * DV + h * DV + col];
        const float gate = sigmoid_gate ? 1.0f / (expf(-zv) + 1.0f) : zv / (1.0f + expf(-zv));
        out[row * hv * DV + h * DV + col] = gdn_round(ry * y * wcol * gate, rd);
    }
    #pragma unroll
    for (int j = 0; j < D; ++j) sp[(size_t)j * DV] = st[j];
}

extern "C" __global__ void __launch_bounds__(128) gdn_mixer_d128_f32(
    const float* __restrict__ qkv, unsigned qkv_stride,
    const float* __restrict__ a, const float* __restrict__ bgate, const float* __restrict__ z,
    const float* __restrict__ A, const float* __restrict__ dt_bias,
    const float* __restrict__ norm_w, float* __restrict__ state, float* __restrict__ out,
    unsigned seq, unsigned hk, unsigned hv, float eps, int sigmoid_gate, int rd)
{
    gdn_mixer_core<128, 128>(qkv, qkv_stride, a, bgate, z, A, dt_bias, norm_w, state, out,
                             seq, hk, hv, eps, sigmoid_gate, rd);
}
