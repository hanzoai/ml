// Fused gated-delta-rule scan (GDN linear-attention recurrence), f32 end-to-end.
// One thread per (batch*head, v_column): a private state column st[kd] in registers, sequential
// over the sequence, no cross-thread communication -- the exact math of hanzo-kernel gdn_scan
// (CPU-gated against the engine's portable scan). q arrives pre-scaled by 1/sqrt(kd).
// Layout: q,k: [bh, seq, kd]  v,out: [bh, seq, vd]  g,beta: [bh, seq]  state: [bh, kd, vd] (in-place).
#include <hip/hip_runtime.h>

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
