// Attention over packed sequences from a packed QKV projection [T, 3·H·64] in bf16.
#include "cuda_utils.cuh"
#include <stdint.h>

typedef __nv_bfloat16 bf16;

#define D 64
#define HALF 32
#define BQ 64
#define BK 64
// A tile's row stride in shared memory.
#define LD 72
#define WARPS 4
#define THREADS (WARPS * 32)

// m16n8k16 at lane l, g = l / 4, c = l % 4: A {(g, 2c), (g+8, 2c), (g, 2c+8), (g+8, 2c+8)}, B {(2c, g), (2c+8, g)}, C {(g, 2c), (g+8, 2c)}.
__device__ __forceinline__ void mma(float *c, const uint32_t *a, const uint32_t *b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t pack(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t *>(&v);
}

__device__ __forceinline__ uint32_t rpair(const bf16 *s, int r, int c0) {
    return *reinterpret_cast<const uint32_t *>(s + r * LD + c0);
}

__device__ __forceinline__ uint32_t cpair(const bf16 *s, int r0, int c) {
    const uint32_t lo = *reinterpret_cast<const uint16_t *>(s + r0 * LD + c);
    const uint32_t hi = *reinterpret_cast<const uint16_t *>(s + (r0 + 1) * LD + c);
    return lo | (hi << 16);
}

// A(i, k) = s[i0 + i][k0 + k]
__device__ __forceinline__ void afrag(uint32_t *a, const bf16 *s, int i0, int k0, int g, int c) {
    a[0] = rpair(s, i0 + g, k0 + 2 * c);
    a[1] = rpair(s, i0 + g + 8, k0 + 2 * c);
    a[2] = rpair(s, i0 + g, k0 + 2 * c + 8);
    a[3] = rpair(s, i0 + g + 8, k0 + 2 * c + 8);
}

// A(i, k) = s[k0 + k][i0 + i]
__device__ __forceinline__ void afragt(uint32_t *a, const bf16 *s, int i0, int k0, int g, int c) {
    a[0] = cpair(s, k0 + 2 * c, i0 + g);
    a[1] = cpair(s, k0 + 2 * c, i0 + g + 8);
    a[2] = cpair(s, k0 + 2 * c + 8, i0 + g);
    a[3] = cpair(s, k0 + 2 * c + 8, i0 + g + 8);
}

// B(k, n) = s[n0 + n][k0 + k]
__device__ __forceinline__ void bfrag(uint32_t *b, const bf16 *s, int n0, int k0, int g, int c) {
    b[0] = rpair(s, n0 + g, k0 + 2 * c);
    b[1] = rpair(s, n0 + g, k0 + 2 * c + 8);
}

// B(k, n) = s[k0 + k][n0 + n]
__device__ __forceinline__ void bfragt(uint32_t *b, const bf16 *s, int n0, int k0, int g, int c) {
    b[0] = cpair(s, k0 + 2 * c, n0 + g);
    b[1] = cpair(s, k0 + 2 * c + 8, n0 + g);
}

// A from the C fragments of two adjacent products.
__device__ __forceinline__ void cfrag(uint32_t *a, const float *c0, const float *c1) {
    a[0] = pack(c0[0], c0[1]);
    a[1] = pack(c0[2], c0[3]);
    a[2] = pack(c1[0], c1[1]);
    a[3] = pack(c1[2], c1[3]);
}

// 64 rows of one head into a tile, zero past `n`; with `rot`, rotated by position and scaled.
__device__ void load(
    bf16 *tile,
    const bf16 *src,
    size_t ld,
    int col,
    int n,
    const bf16 *cos,
    const bf16 *sin,
    int pos0,
    bool rot,
    float mul
) {
    for (int idx = threadIdx.x; idx < BQ * HALF; idx += THREADS) {
        const int r = idx / HALF, d = idx % HALF;
        float x1 = 0.f, x2 = 0.f;
        if (r < n) {
            const bf16 *row = src + (size_t)r * ld + col;
            x1 = __bfloat162float(row[d]);
            x2 = __bfloat162float(row[d + HALF]);
            if (rot) {
                const float c = __bfloat162float(cos[(size_t)(pos0 + r) * HALF + d]);
                const float s = __bfloat162float(sin[(size_t)(pos0 + r) * HALF + d]);
                const float y1 = (x1 * c - x2 * s) * mul;
                const float y2 = (x1 * s + x2 * c) * mul;
                x1 = y1;
                x2 = y2;
            }
        }
        tile[r * LD + d] = __float2bfloat16(x1);
        tile[r * LD + d + HALF] = __float2bfloat16(x2);
    }
}

__device__ __forceinline__ float qmax(float x) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, 1, 32));
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, 2, 32));
    return x;
}

__device__ __forceinline__ float qsum(float x) {
    x += __shfl_xor_sync(0xffffffff, x, 1, 32);
    x += __shfl_xor_sync(0xffffffff, x, 2, 32);
    return x;
}

__device__ __forceinline__ bool reads(int i, int j, int len, int window) {
    return i < len && j < len && (window < 0 || abs(i - j) <= window);
}

extern "C" __global__ void __launch_bounds__(THREADS) flash_fwd_bf16(
    const bf16 *qkv,
    const bf16 *cos,
    const bf16 *sin,
    const uint32_t *cu,
    const uint32_t *tiles,
    const uint32_t heads,
    const int32_t window,
    const float scale,
    bf16 *out,
    float *lse,
    const uint32_t total
) {
    __shared__ __align__(16) bf16 qs[BQ * LD];
    __shared__ __align__(16) bf16 ks[BK * LD];
    __shared__ __align__(16) bf16 vs[BK * LD];
    const int h = blockIdx.y, hd = heads * D;
    const int seq = tiles[2 * blockIdx.x], i0 = tiles[2 * blockIdx.x + 1];
    const int start = cu[seq], len = cu[seq + 1] - start;
    const size_t ld = 3 * (size_t)hd;
    const bf16 *base = qkv + (size_t)start * ld;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, c = lane % 4;
    const int nq = min(BQ, len - i0);

    load(qs, base + (size_t)i0 * ld, ld, h * D, nq, cos, sin, i0, true, scale);
    __syncthreads();
    uint32_t qa[4][4];
#pragma unroll
    for (int kc = 0; kc < 4; kc++) {
        afrag(qa[kc], qs, warp * 16, kc * 16, g, c);
    }
    float o[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.f;
    }
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    const int iq[2] = {i0 + warp * 16 + g, i0 + warp * 16 + g + 8};

    int j_lo = 0, j_hi = len;
    if (window >= 0) {
        j_lo = max(0, i0 - window);
        j_hi = min(len, i0 + nq + window);
    }
    j_lo &= ~(BK - 1);
    for (int j0 = j_lo; j0 < j_hi; j0 += BK) {
        const int nk = min(BK, len - j0);
        __syncthreads();
        load(ks, base + (size_t)j0 * ld, ld, hd + h * D, nk, cos, sin, j0, true, 1.f);
        load(vs, base + (size_t)j0 * ld, ld, 2 * hd + h * D, nk, cos, sin, 0, false, 1.f);
        __syncthreads();

        float s[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
            for (int kc = 0; kc < 4; kc++) {
                uint32_t b[2];
                bfrag(b, ks, nt * 8, kc * 16, g, c);
                mma(s[nt], qa[kc], b);
            }
        }
        float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int row = e / 2, j = j0 + nt * 8 + 2 * c + (e % 2);
                if (!reads(iq[row], j, len, window)) {
                    s[nt][e] = -INFINITY;
                }
                mx[row] = fmaxf(mx[row], s[nt][e]);
            }
        }
        float alpha[2];
#pragma unroll
        for (int row = 0; row < 2; row++) {
            const float mn = fmaxf(m[row], qmax(mx[row]));
            alpha[row] = mn == -INFINITY ? 1.f : expf(m[row] - mn);
            m[row] = mn;
            l[row] *= alpha[row];
        }
        float rs[2] = {0.f, 0.f};
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int row = e / 2;
                const float p = s[nt][e] == -INFINITY ? 0.f : expf(s[nt][e] - m[row]);
                s[nt][e] = p;
                rs[row] += p;
                o[nt][e] *= alpha[row];
            }
        }
#pragma unroll
        for (int row = 0; row < 2; row++) {
            l[row] += qsum(rs[row]);
        }
#pragma unroll
        for (int kc = 0; kc < 4; kc++) {
            uint32_t pa[4];
            cfrag(pa, s[2 * kc], s[2 * kc + 1]);
#pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                uint32_t b[2];
                bfragt(b, vs, nt * 8, kc * 16, g, c);
                mma(o[nt], pa, b);
            }
        }
    }

#pragma unroll
    for (int row = 0; row < 2; row++) {
        const int i = iq[row];
        if (i >= len) {
            continue;
        }
        const float inv = 1.f / l[row];
        bf16 *dst = out + (size_t)(start + i) * hd + h * D;
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            *reinterpret_cast<uint32_t *>(dst + nt * 8 + 2 * c) =
                pack(o[nt][2 * row] * inv, o[nt][2 * row + 1] * inv);
        }
        if (c == 0) {
            lse[(size_t)h * total + start + i] = m[row] + logf(l[row]);
        }
    }
}

extern "C" __global__ void flash_delta_bf16(
    const bf16 *o,
    const bf16 *dout,
    const uint32_t heads,
    const uint32_t total,
    float *delta
) {
    const size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= (size_t)total * heads) {
        return;
    }
    const size_t t = idx / heads, h = idx % heads;
    const bf16 *a = o + (t * heads + h) * D;
    const bf16 *b = dout + (t * heads + h) * D;
    float acc = 0.f;
#pragma unroll
    for (int d = 0; d < D; d += 2) {
        const __nv_bfloat162 x = *reinterpret_cast<const __nv_bfloat162 *>(a + d);
        const __nv_bfloat162 y = *reinterpret_cast<const __nv_bfloat162 *>(b + d);
        acc += __bfloat162float(x.x) * __bfloat162float(y.x)
             + __bfloat162float(x.y) * __bfloat162float(y.y);
    }
    delta[h * (size_t)total + t] = acc;
}

extern "C" __global__ void __launch_bounds__(THREADS) flash_bwd_bf16(
    const bf16 *qkv,
    const bf16 *cos,
    const bf16 *sin,
    const uint32_t *cu,
    const uint32_t *tiles,
    const uint32_t heads,
    const int32_t window,
    const float scale,
    const bf16 *dout,
    const float *lse,
    const float *delta,
    float *dq,
    bf16 *dqkv,
    const uint32_t total
) {
    __shared__ __align__(16) bf16 ks[BK * LD];
    __shared__ __align__(16) bf16 vs[BK * LD];
    __shared__ __align__(16) bf16 qs[BQ * LD];
    __shared__ __align__(16) bf16 dos[BQ * LD];
    __shared__ __align__(16) bf16 dss[BK * LD];
    __shared__ float lses[BQ], deltas[BQ];
    const int h = blockIdx.y, hd = heads * D;
    const int seq = tiles[2 * blockIdx.x], j0 = tiles[2 * blockIdx.x + 1];
    const int start = cu[seq], len = cu[seq + 1] - start;
    const size_t ld = 3 * (size_t)hd;
    const bf16 *base = qkv + (size_t)start * ld;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, c = lane % 4;
    const int nk = min(BK, len - j0);

    load(ks, base + (size_t)j0 * ld, ld, hd + h * D, nk, cos, sin, j0, true, 1.f);
    load(vs, base + (size_t)j0 * ld, ld, 2 * hd + h * D, nk, cos, sin, 0, false, 1.f);
    __syncthreads();
    uint32_t ka[4][4], va[4][4];
#pragma unroll
    for (int kc = 0; kc < 4; kc++) {
        afrag(ka[kc], ks, warp * 16, kc * 16, g, c);
        afrag(va[kc], vs, warp * 16, kc * 16, g, c);
    }
    float dk[8][4], dv[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        dk[nt][0] = dk[nt][1] = dk[nt][2] = dk[nt][3] = 0.f;
        dv[nt][0] = dv[nt][1] = dv[nt][2] = dv[nt][3] = 0.f;
    }
    const int jk[2] = {j0 + warp * 16 + g, j0 + warp * 16 + g + 8};

    int i_lo = 0, i_hi = len;
    if (window >= 0) {
        i_lo = max(0, j0 - window);
        i_hi = min(len, j0 + nk + window);
    }
    i_lo &= ~(BQ - 1);
    for (int i0 = i_lo; i0 < i_hi; i0 += BQ) {
        const int nq = min(BQ, len - i0);
        __syncthreads();
        load(qs, base + (size_t)i0 * ld, ld, h * D, nq, cos, sin, i0, true, scale);
        load(dos, dout + (size_t)(start + i0) * hd, hd, h * D, nq, cos, sin, 0, false, 1.f);
        if (threadIdx.x < BQ) {
            const bool in = (int)threadIdx.x < nq;
            const size_t at = (size_t)h * total + start + i0 + threadIdx.x;
            lses[threadIdx.x] = in ? lse[at] : 0.f;
            deltas[threadIdx.x] = in ? delta[at] : 0.f;
        }
        __syncthreads();

        float st[8][4], dpt[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            st[nt][0] = st[nt][1] = st[nt][2] = st[nt][3] = 0.f;
            dpt[nt][0] = dpt[nt][1] = dpt[nt][2] = dpt[nt][3] = 0.f;
#pragma unroll
            for (int kc = 0; kc < 4; kc++) {
                uint32_t b[2];
                bfrag(b, qs, nt * 8, kc * 16, g, c);
                mma(st[nt], ka[kc], b);
                bfrag(b, dos, nt * 8, kc * 16, g, c);
                mma(dpt[nt], va[kc], b);
            }
        }
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int row = e / 2, q = nt * 8 + 2 * c + (e % 2), i = i0 + q;
                const float p = reads(i, jk[row], len, window) ? expf(st[nt][e] - lses[q]) : 0.f;
                st[nt][e] = p;
                dpt[nt][e] = p * (dpt[nt][e] - deltas[q]);
            }
        }
#pragma unroll
        for (int kc = 0; kc < 4; kc++) {
            uint32_t pa[4], da[4];
            cfrag(pa, st[2 * kc], st[2 * kc + 1]);
            cfrag(da, dpt[2 * kc], dpt[2 * kc + 1]);
#pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                uint32_t b[2];
                bfragt(b, dos, nt * 8, kc * 16, g, c);
                mma(dv[nt], pa, b);
                bfragt(b, qs, nt * 8, kc * 16, g, c);
                mma(dk[nt], da, b);
            }
        }
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            *reinterpret_cast<uint32_t *>(dss + (warp * 16 + g) * LD + nt * 8 + 2 * c) =
                pack(dpt[nt][0], dpt[nt][1]);
            *reinterpret_cast<uint32_t *>(dss + (warp * 16 + g + 8) * LD + nt * 8 + 2 * c) =
                pack(dpt[nt][2], dpt[nt][3]);
        }
        __syncthreads();
        float dqa[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            dqa[nt][0] = dqa[nt][1] = dqa[nt][2] = dqa[nt][3] = 0.f;
        }
#pragma unroll
        for (int kc = 0; kc < 4; kc++) {
            uint32_t a[4];
            afragt(a, dss, warp * 16, kc * 16, g, c);
#pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                uint32_t b[2];
                bfragt(b, ks, nt * 8, kc * 16, g, c);
                mma(dqa[nt], a, b);
            }
        }
#pragma unroll
        for (int row = 0; row < 2; row++) {
            const int i = i0 + warp * 16 + g + 8 * row;
            if (i >= len) {
                continue;
            }
            float *dst = dq + (size_t)(start + i) * hd + h * D;
#pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                atomicAdd(dst + nt * 8 + 2 * c, dqa[nt][2 * row]);
                atomicAdd(dst + nt * 8 + 2 * c + 1, dqa[nt][2 * row + 1]);
            }
        }
    }

#pragma unroll
    for (int row = 0; row < 2; row++) {
        const int j = jk[row];
        if (j >= len) {
            continue;
        }
        bf16 *dst = dqkv + (size_t)(start + j) * ld + h * D;
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int d = nt * 8 + 2 * c + e;
                const float x1 = dk[nt][2 * row + e], x2 = dk[nt + 4][2 * row + e];
                const float cs = __bfloat162float(cos[(size_t)j * HALF + d]);
                const float sn = __bfloat162float(sin[(size_t)j * HALF + d]);
                dst[hd + d] = __float2bfloat16(x1 * cs + x2 * sn);
                dst[hd + d + HALF] = __float2bfloat16(x2 * cs - x1 * sn);
            }
        }
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            *reinterpret_cast<uint32_t *>(dst + 2 * hd + nt * 8 + 2 * c) =
                pack(dv[nt][2 * row], dv[nt][2 * row + 1]);
        }
    }
}

extern "C" __global__ void flash_dq_bf16(
    const float *dq,
    const bf16 *cos,
    const bf16 *sin,
    const uint32_t *cu,
    const uint32_t seqs,
    const uint32_t heads,
    const float scale,
    const uint32_t total,
    bf16 *dqkv
) {
    const size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= (size_t)total * heads * HALF) {
        return;
    }
    const int d = idx % HALF;
    const size_t th = idx / HALF;
    const int h = th % heads;
    const size_t t = th / heads;
    uint32_t lo = 0, hi = seqs;
    while (hi - lo > 1) {
        const uint32_t mid = (lo + hi) / 2;
        if (cu[mid] <= t) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const size_t pos = t - cu[lo];
    const float *src = dq + (t * heads + h) * D;
    const float x1 = src[d], x2 = src[d + HALF];
    const float cs = __bfloat162float(cos[pos * HALF + d]);
    const float sn = __bfloat162float(sin[pos * HALF + d]);
    bf16 *dst = dqkv + t * 3 * (size_t)heads * D + (size_t)h * D;
    dst[d] = __float2bfloat16(scale * (x1 * cs + x2 * sn));
    dst[d + HALF] = __float2bfloat16(scale * (x2 * cs - x1 * sn));
}
