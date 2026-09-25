#include <metal_stdlib>
using namespace metal;

// One AdamW step over a contiguous F32 parameter, in place, in one pass over memory:
//   g <- s g
//   m <- b1 m + (1 - b1) g
//   v <- b2 v + (1 - b2) g^2
//   w <- w (1 - lr wd) - lr (m cm) / (sqrt(v cv) + eps)
// cm = 1 / (1 - b1^t) and cv = 1 / (1 - b2^t) are the bias corrections, s scales the gradient
// (clipping).
struct AdamW {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float scale_m;
    float scale_v;
    float grad_scale;
};

kernel void adamw_f32(
    constant uint &n,
    constant AdamW &p,
    device float *w,
    device const float *g,
    device float *m,
    device float *v,
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) {
        return;
    }
    const float gi = g[tid] * p.grad_scale;
    const float mi = p.beta1 * m[tid] + (1.0f - p.beta1) * gi;
    const float vi = p.beta2 * v[tid] + (1.0f - p.beta2) * gi * gi;
    m[tid] = mi;
    v[tid] = vi;
    const float update = (mi * p.scale_m) / (sqrt(vi * p.scale_v) + p.eps);
    w[tid] = w[tid] * (1.0f - p.lr * p.weight_decay) - p.lr * update;
}

// Sum of squares of a contiguous F32 buffer, accumulated into out[0] with an atomic add; one
// partial per threadgroup. A gradient's squared norm without a reduction tree per tensor.
kernel void sumsq_f32(
    constant uint &n,
    device const float *x,
    device atomic_float *out,
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint grid [[threads_per_grid]]
) {
    threadgroup float partial[32];
    float acc = 0.0f;
    for (uint i = tid; i < n; i += grid) {
        const float xi = x[i];
        acc += xi * xi;
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        partial[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const uint groups = (tg_size + 31) / 32;
        float total = lane < groups ? partial[lane] : 0.0f;
        total = simd_sum(total);
        if (lane == 0) {
            atomic_fetch_add_explicit(out, total, memory_order_relaxed);
        }
    }
}
