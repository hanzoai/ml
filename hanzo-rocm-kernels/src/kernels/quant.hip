// ROCm/HIP native quantized matvec kernels (decode path).
// Reads the GGML quantized block format straight from VRAM -- no CPU dequant, no re-pack --
// so the result is exact w.r.t. the CPU reference. This is the bandwidth lever for memory-bound
// decode: weights stay quantized (Q8_0 ~1.06 B/elem) instead of being expanded to dense f16.

#ifndef __HIPCC__
#define __device__
#define __global__
#else
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#endif

#include <stddef.h>
#include <stdint.h>

// Quantize activations to symmetric int8 with a per-32-block f16 scale (q8_0-style on the
// activation side, as llama's quantize_q8_1 does; the q8_1 sum term is unused vs symmetric Q8_0
// weights). x[M,K] f16 -> xq[M,K] int8 + xd[M, K/32] f16. One warp per (row, 32-block).
extern "C" __global__ void quantize_q8(
    const int M,
    const int K,
    const __half* __restrict__ x,
    int8_t* __restrict__ xq,
    __half* __restrict__ xd
) {
    const int nblk = K >> 5;
    const int wid = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (wid >= M * nblk) {
        return;
    }
    const int m = wid / nblk;
    const int blk = wid % nblk;
    const int lane = threadIdx.x & 31;
    const size_t idx = (size_t)m * K + (size_t)blk * 32 + lane;
    const float v = __half2float(x[idx]);
    float a = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a = fmaxf(a, __shfl_xor(a, off)); // warp max-reduce -> every lane has the block absmax
    }
    const float inv = (a > 0.0f) ? (127.0f / a) : 0.0f;
    // roundf = round-half-AWAY-from-zero, matching llama's quantize_mmq_q8_1 (quantize.cu) and the
    // CPU reference in qmmq_numeric.rs (Rust f32::round). rintf (round-half-to-even) differed from
    // both on exact .5 ties; roundf removes that 1-ULP divergence vs llama at no perf cost.
    int q = (int)roundf(v * inv);
    q = max(-127, min(127, q));
    xq[idx] = (int8_t)q;
    if (lane == 0) {
        xd[(size_t)m * nblk + blk] = __float2half(a / 127.0f);
    }
}

// Q8_1-style activation quant: identical int8 `xq` + per-32-block f16 scale `xd` as `quantize_q8`
// above, PLUS the per-32-block int8 SUM `xs[m,blk] = sum_k q(k)` (llama's block_q8_1 `s` field).
// The sum is the bias term for ASYMMETRIC weights (Q4_K/Q5_K), where the dequant carries a per-
// sub-block min: out -= dmin_w * m_g * d_x * sum_k(q_x). Symmetric weights never read `xs`. Keeping
// this a SEPARATE kernel means the proven Q8_0/Q4_0 prefill (which calls `quantize_q8`) is byte-
// unchanged; only the asymmetric launcher pays for the extra sum reduction + store.
extern "C" __global__ void quantize_q8_1(
    const int M,
    const int K,
    const __half* __restrict__ x,
    int8_t* __restrict__ xq,
    __half* __restrict__ xd,
    int* __restrict__ xs        // [M, K/32] per-block int8 sum (i32)
) {
    const int nblk = K >> 5;
    const int wid = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (wid >= M * nblk) {
        return;
    }
    const int m = wid / nblk;
    const int blk = wid % nblk;
    const int lane = threadIdx.x & 31;
    const size_t idx = (size_t)m * K + (size_t)blk * 32 + lane;
    const float v = __half2float(x[idx]);
    float a = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a = fmaxf(a, __shfl_xor(a, off));
    }
    const float inv = (a > 0.0f) ? (127.0f / a) : 0.0f;
    int q = (int)roundf(v * inv);
    q = max(-127, min(127, q));
    xq[idx] = (int8_t)q;
    // Warp sum of the int8 quants for this block (the q8_1 sum term).
    int qsum = q;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        qsum += __shfl_xor(qsum, off);
    }
    if (lane == 0) {
        xd[(size_t)m * nblk + blk] = __float2half(a / 127.0f);
        xs[(size_t)m * nblk + blk] = qsum;
    }
}

// bf16 activation variant of quantize_q8_1: byte-identical q8_1 quant (int8 xq + per-32-block f16
// scale xd + int sum xs), reading bf16 instead of f16. The decode path keeps the model's working
// bf16 dtype, so the dp4a Q4_K matvec needs a bf16-input q8_1 quant (the f16 one above rejects bf16);
// the only difference is the activation load (hip_bfloat16 -> float), the quant math is the same.
extern "C" __global__ void quantize_q8_1_bf16(
    const int M,
    const int K,
    const hip_bfloat16* __restrict__ x,
    int8_t* __restrict__ xq,
    __half* __restrict__ xd,
    int* __restrict__ xs
) {
    const int nblk = K >> 5;
    const int wid = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (wid >= M * nblk) {
        return;
    }
    const int m = wid / nblk;
    const int blk = wid % nblk;
    const int lane = threadIdx.x & 31;
    const size_t idx = (size_t)m * K + (size_t)blk * 32 + lane;
    const float v = (float)x[idx];
    float a = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a = fmaxf(a, __shfl_xor(a, off));
    }
    const float inv = (a > 0.0f) ? (127.0f / a) : 0.0f;
    int q = (int)roundf(v * inv);
    q = max(-127, min(127, q));
    xq[idx] = (int8_t)q;
    int qsum = q;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        qsum += __shfl_xor(qsum, off);
    }
    if (lane == 0) {
        xd[(size_t)m * nblk + blk] = __float2half(a / 127.0f);
        xs[(size_t)m * nblk + blk] = qsum;
    }
}

// get_scale_min_k4 scale/min unpack -- DEFINED below (reused by the scalar Q4_K path + the unified
// core); forward-declared here so the dp4a Q4_K kernel can call the SAME unpack (one source of
// truth for the 6-bit scale layout).
__device__ __forceinline__ void q4k_scale_min(const uint8_t* __restrict__ s, int j, int* sc, int* m);

// ----------------------------------------------------------------------------------------------
// dp4a (4-element signed int8 dot-accumulate) for the int8 decode path. FAITHFUL to llama.cpp's
// ggml_cuda_dp4a (common.cuh): on RDNA3/RDNA3.5/RDNA4 the signed-int8 v_dot4 is exposed as
// __builtin_amdgcn_sudot4(neg_a_unsigned=?, a, neg_b_unsigned=?, b, c, clamp); llama passes
// (true, a, true, b, c, false) -- the two `true`s select SIGNED operands (NOT unsigned), giving a
// signed*signed 4xint8 dot accumulated into c. gfx1151 (RDNA3.5) does NOT have the `dot1-insts`
// feature that `__builtin_amdgcn_sdot4` needs (verified: sdot4 fails to compile here), so sudot4 is
// the correct builtin -- exactly llama's RDNA3 path. Returns c + sum_{k=0..3} a8[k]*b8[k].
__device__ __forceinline__ int hip_dp4a(int a, int b, int c) {
#if defined(__HIPCC__)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
    const int8_t* a8 = reinterpret_cast<const int8_t*>(&a);
    const int8_t* b8 = reinterpret_cast<const int8_t*>(&b);
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
}

// ----------------------------------------------------------------------------------------------
// Native Q4_K decode matvec via int8 dp4a -- FAITHFUL port of llama.cpp's vec_dot_q4_K_q8_1 +
// vec_dot_q4_K_q8_1_impl_vmmq (ggml-cuda/vecdotq.cuh). This REPLACES the scalar-float qmatvec_q4k_*
// path (which dequantized to f32 and did float MACs) with the same int8 SIMD dot llama.cpp runs:
// the activation row is pre-quantized to q8_1 (int8 qs + per-32-block f16 scale d8), and each weight
// nibble is dotted against the q8_1 int8 directly via v_dot4 (4x int8 MAC/instruction), then scaled.
//
// MATH (per Q4_K super-block: 256 weights = 8 sub-blocks of 32; block_q4_K = {d, dmin, 12 scale
// bytes, 128 qs bytes}). For sub-block j in [0,8): (sc[j], m[j]) = get_scale_min_k4(scales, j);
// the 32 weights are nibbles of qs -- chunk c = j/2 covers qs[c*32, c*32+32); sub-block 2c uses the
// LOW nibble of each byte, sub-block 2c+1 the HIGH nibble (exactly llama's v0i=(v>>0)&0x0F0F0F0F /
// v1i=(v>>4)&0x0F0F0F0F split and the to_float chunk layout). The matching q8_1 block j has int8
// quants u[j][0..31] (as 8 ints) and scale d8[j]. Then, identical to impl_vmmq:
//     dot1 = sum_k nibble_k * u_k        (8x dp4a over the 32 elems)
//     dot2 = sum_k u_k                   (8x dp4a of 0x01010101 vs u  -- sum of q8_1 quants)
//     sumf_d += d8[j] * (dot1 * sc[j])
//     sumf_m += d8[j] * (dot2 * m[j])
//     result += d * sumf_d - dmin * sumf_m
// llama tiles the 8 sub-blocks across the mmvq thread grid via the iqs/bq8_offset machinery and 4
// vec_dot calls of QR4_K=2 each; summing all 8 sub-blocks in one loop is algebraically identical
// (same per-element products, same f32 accumulation of the d*sc*dot1 - dmin*m*dot2 terms).
//
// WARP-per-row, LANE-STRIDED over super-blocks (matches the proven unified decode core): lane L
// streams whole super-blocks b = L, L+32, ... so 32 independent super-block loads are in flight
// (hides LPDDR5X latency on the m=1 FFN shapes), each lane fully int8-decodes its super-blocks, then
// a warp-shuffle reduces the per-lane partials. The q8_1 activation (xq int8 + xd f16 scale) is laid
// out [k] contiguous: super-block b, sub-block j -> q8 ints at xq + (b*8 + j)*32, scale xd[b*8 + j].

#define Q4K_BLOCK_BYTES_DP4A 144

// Lane's full int8 dot of ONE Q4_K super-block against its 8 q8_1 activation blocks. `blk` -> the
// 144-byte Q4_K super-block; `xq8` -> the 256 int8 q8_1 quants for this super-block (8 blocks of 32,
// contiguous); `xd8` -> the 8 per-32-block f16 q8_1 scales for this super-block.
__device__ __forceinline__ float q4k_dp4a_block(
        const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
    const float d    = __half2float(*reinterpret_cast<const __half*>(blk));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(blk + 2));
    const uint8_t* scales = blk + 4;        // 12 packed 6-bit scale bytes
    const int* q4 = reinterpret_cast<const int*>(blk + 16);  // 128 qs bytes = 32 ints
    const int* u  = reinterpret_cast<const int*>(xq8);       // 256 int8 = 64 ints (8 blocks * 8 ints)

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
    #pragma unroll
    for (int c = 0; c < 4; ++c) {        // 4 chunks, each = qs[c*32, c*32+32) = 8 ints
        // Sub-block 2c (low nibbles) and 2c+1 (high nibbles) share the same 8 weight ints.
        int sc_lo, m_lo, sc_hi, m_hi;
        q4k_scale_min(scales, 2 * c,     &sc_lo, &m_lo);
        q4k_scale_min(scales, 2 * c + 1, &sc_hi, &m_hi);
        const int* q4c   = q4 + c * 8;          // 8 ints of this chunk's nibble-packed weights
        const int* u_lo  = u + (2 * c)     * 8; // q8_1 block 2c   (8 ints)
        const int* u_hi  = u + (2 * c + 1) * 8; // q8_1 block 2c+1 (8 ints)
        const float d8_lo = __half2float(xd8[2 * c]);
        const float d8_hi = __half2float(xd8[2 * c + 1]);
        int dot1_lo = 0, dot2_lo = 0, dot1_hi = 0, dot2_hi = 0;
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int w   = q4c[t];
            const int wlo = (w >> 0) & 0x0F0F0F0F;   // low  nibble of each byte
            const int whi = (w >> 4) & 0x0F0F0F0F;   // high nibble of each byte
            dot1_lo = hip_dp4a(wlo, u_lo[t], dot1_lo);
            dot2_lo = hip_dp4a(0x01010101, u_lo[t], dot2_lo);
            dot1_hi = hip_dp4a(whi, u_hi[t], dot1_hi);
            dot2_hi = hip_dp4a(0x01010101, u_hi[t], dot2_hi);
        }
        sumf_d += d8_lo * (dot1_lo * sc_lo) + d8_hi * (dot1_hi * sc_hi);
        sumf_m += d8_lo * (dot2_lo * m_lo)  + d8_hi * (dot2_hi * m_hi);
    }
    return d * sumf_d - dmin * sumf_m;
}

// f16 activation/output. wq = raw Q4_K weight bytes; xq/xd = the pre-quantized q8_1 activation row.
extern "C" __global__ void qmatvec_q4k_dp4a_f16(
    const int nrows,
    const int ncols,            // multiple of 256
    const uint8_t* __restrict__ wq,
    const int8_t*  __restrict__ xq,   // [ncols] q8_1 int8 quants
    const __half*  __restrict__ xd,   // [ncols/32] q8_1 f16 scales
    __half* __restrict__ y
) {
    const int lane = threadIdx.x & 31;
    const int rows_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * rows_per_block + (threadIdx.x >> 5);
    if (row >= nrows) {
        return;
    }
    const int nblocks = ncols >> 8; // ncols / 256
    const uint8_t* row_ptr = wq + (size_t)row * (size_t)nblocks * Q4K_BLOCK_BYTES_DP4A;

    float acc = 0.0f;
    for (int b = lane; b < nblocks; b += 32) {
        const uint8_t* blk = row_ptr + (size_t)b * Q4K_BLOCK_BYTES_DP4A;
        const int8_t* xq8  = xq + (size_t)b * 256;
        const __half* xd8  = xd + (size_t)b * 8;
        acc += q4k_dp4a_block(blk, xq8, xd8);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off);
    }
    if (lane == 0) {
        y[row] = __float2half(acc);
    }
}

// bf16 activation/output (model working dtype). Identical int8 math; only the output store differs
// (the q8_1 activation is already int8 + f16 scale, dtype-independent, produced by quantize_q8_1).
extern "C" __global__ void qmatvec_q4k_dp4a_bf16(
    const int nrows,
    const int ncols,            // multiple of 256
    const uint8_t* __restrict__ wq,
    const int8_t*  __restrict__ xq,
    const __half*  __restrict__ xd,
    hip_bfloat16* __restrict__ y
) {
    const int lane = threadIdx.x & 31;
    const int rows_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * rows_per_block + (threadIdx.x >> 5);
    if (row >= nrows) {
        return;
    }
    const int nblocks = ncols >> 8; // ncols / 256
    const uint8_t* row_ptr = wq + (size_t)row * (size_t)nblocks * Q4K_BLOCK_BYTES_DP4A;

    float acc = 0.0f;
    for (int b = lane; b < nblocks; b += 32) {
        const uint8_t* blk = row_ptr + (size_t)b * Q4K_BLOCK_BYTES_DP4A;
        const int8_t* xq8  = xq + (size_t)b * 256;
        const __half* xd8  = xd + (size_t)b * 8;
        acc += q4k_dp4a_block(blk, xq8, xd8);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off);
    }
    if (lane == 0) {
        y[row] = hip_bfloat16(acc);
    }
}

// Batched indexed-MoE Q4_K decode matvec via int8 dp4a. Same per-row math as qmatvec_q4k_dp4a_*
// above, but ALL routed slots run in ONE launch with experts on grid.y: for slot s = blockIdx.y,
// expert = ids[s], and the warp computes output row `row` of that expert's [n,k] weight against
// slot s's pre-quantized q8_1 activation (xq + s*ncols, xd + s*ncols/32) into y[s*n + row]. This
// collapses the host per-expert launch loop (topk tiny launches) into one well-occupied grid AND
// uses the dp4a int8 core instead of scalar-float dequant -- the two levers that bring MoE decode
// to the same roofline as the non-MoE dp4a matvec. `wbank` = [E,n,k] resident GGML Q4_K bytes.
extern "C" __global__ void moe_qmatvec_q4k_dp4a_f16(
    const int n,                        // output rows per expert (weight rows)
    const int ncols,                    // k, multiple of 256
    const int nslots,                   // routed slots (= nrows)
    const uint8_t* __restrict__ wbank,  // [E, n, k] Q4_K blocks
    const int* __restrict__ ids,        // [nslots] expert id per slot
    const int8_t* __restrict__ xq,      // [nslots, ncols] q8_1 int8 quants
    const __half* __restrict__ xd,      // [nslots, ncols/32] q8_1 f16 scales
    __half* __restrict__ y              // [nslots, n]
) {
    const int s = blockIdx.y;
    if (s >= nslots) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const int rows_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * rows_per_block + (threadIdx.x >> 5);
    if (row >= n) {
        return;
    }
    const int nblocks = ncols >> 8; // ncols / 256
    const int expert = ids[s];
    const uint8_t* row_ptr =
        wbank + ((size_t)expert * n + row) * (size_t)nblocks * Q4K_BLOCK_BYTES_DP4A;
    const int8_t* xq_row = xq + (size_t)s * ncols;
    const __half* xd_row = xd + (size_t)s * (ncols >> 5);

    float acc = 0.0f;
    for (int b = lane; b < nblocks; b += 32) {
        const uint8_t* blk = row_ptr + (size_t)b * Q4K_BLOCK_BYTES_DP4A;
        const int8_t* xq8  = xq_row + (size_t)b * 256;
        const __half* xd8  = xd_row + (size_t)b * 8;
        acc += q4k_dp4a_block(blk, xq8, xd8);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off);
    }
    if (lane == 0) {
        y[(size_t)s * n + row] = __float2half(acc);
    }
}

// bf16 activation/output mirror of moe_qmatvec_q4k_dp4a_f16 (model working dtype). Identical int8
// math; only the output store type differs (the q8_1 activation is dtype-independent).
extern "C" __global__ void moe_qmatvec_q4k_dp4a_bf16(
    const int n,
    const int ncols,
    const int nslots,
    const uint8_t* __restrict__ wbank,
    const int* __restrict__ ids,
    const int8_t* __restrict__ xq,
    const __half* __restrict__ xd,
    hip_bfloat16* __restrict__ y
) {
    const int s = blockIdx.y;
    if (s >= nslots) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const int rows_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * rows_per_block + (threadIdx.x >> 5);
    if (row >= n) {
        return;
    }
    const int nblocks = ncols >> 8;
    const int expert = ids[s];
    const uint8_t* row_ptr =
        wbank + ((size_t)expert * n + row) * (size_t)nblocks * Q4K_BLOCK_BYTES_DP4A;
    const int8_t* xq_row = xq + (size_t)s * ncols;
    const __half* xd_row = xd + (size_t)s * (ncols >> 5);

    float acc = 0.0f;
    for (int b = lane; b < nblocks; b += 32) {
        const uint8_t* blk = row_ptr + (size_t)b * Q4K_BLOCK_BYTES_DP4A;
        const int8_t* xq8  = xq_row + (size_t)b * 256;
        const __half* xd8  = xd_row + (size_t)b * 8;
        acc += q4k_dp4a_block(blk, xq8, xd8);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off);
    }
    if (lane == 0) {
        y[(size_t)s * n + row] = hip_bfloat16(acc);
    }
}

// Q8_0 block layout (GGML): 2-byte f16 scale `d`, then 32 int8 quants. 34 bytes, 32 weights.
// Weight matrix W is [nrows, ncols] row-major; row r holds ncols/32 consecutive Q8_0 blocks.
//   y[r] = sum_k dequant(W[r,k]) * x[k],   dequant(W[r,k]) = d_block * qs[k_in_block]
//
// One WARP per output row (blockDim/32 rows per block, so all lanes stay busy even when a row has
// fewer Q8_0 blocks than 256 -- the old one-block-per-row left half the threads idle at k=4096).
// Each lane owns whole Q8_0 blocks (the f16 scale is read once per 32 MACs and the inner 32-wide
// loop vectorizes); a warp covers 32 contiguous blocks per step, then a warp-shuffle reduction
// (no shared memory, no __syncthreads).
extern "C" __global__ void qmatvec_q8_0_f16(
    const int nrows,
    const int ncols,            // multiple of 32
    const uint8_t* __restrict__ wq,
    const __half* __restrict__ x,
    __half* __restrict__ y
) {
    const int lane = threadIdx.x & 31;
    const int rows_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * rows_per_block + (threadIdx.x >> 5);
    if (row >= nrows) {
        return;
    }
    const int nblocks = ncols >> 5; // ncols / 32
    const uint8_t* row_ptr = wq + (size_t)row * (size_t)nblocks * 34;

    float acc = 0.0f;
    for (int b = lane; b < nblocks; b += 32) {
        const uint8_t* blk = row_ptr + (size_t)b * 34;
        // blk is 2-byte aligned (34 is even, base is device-aligned), so the f16 read is aligned.
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
        const __half* xb = x + (size_t)b * 32;
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            s += (float)qs[i] * __half2float(xb[i]);
        }
        acc += d * s;
    }

    // Warp reduction (wave32).
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off);
    }
    if (lane == 0) {
        y[row] = __float2half(acc);
    }
}

// BF16-native decode matvec: byte-identical math to qmatvec_q8_0_f16, but reads the activation
// straight as bf16 and writes bf16 -- so the decode path no longer round-trips bf16->f32->f16->bf16
// (3 cast launches per matvec). The weight dequant + f32 accumulation are unchanged; only the
// activation/output element type differs (hip_bfloat16 has an implicit float conversion both ways).
extern "C" __global__ void qmatvec_q8_0_bf16(
    const int nrows,
    const int ncols,            // multiple of 32
    const uint8_t* __restrict__ wq,
    const hip_bfloat16* __restrict__ x,
    hip_bfloat16* __restrict__ y
) {
    const int lane = threadIdx.x & 31;
    const int rows_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * rows_per_block + (threadIdx.x >> 5);
    if (row >= nrows) {
        return;
    }
    const int nblocks = ncols >> 5; // ncols / 32
    const uint8_t* row_ptr = wq + (size_t)row * (size_t)nblocks * 34;

    float acc = 0.0f;
    for (int b = lane; b < nblocks; b += 32) {
        const uint8_t* blk = row_ptr + (size_t)b * 34;
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
        const hip_bfloat16* xb = x + (size_t)b * 32;
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            s += (float)qs[i] * (float)xb[i];
        }
        acc += d * s;
    }

    // Warp reduction (wave32).
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off);
    }
    if (lane == 0) {
        y[row] = hip_bfloat16(acc);
    }
}

// ----------------------------------------------------------------------------------------------
// Native Q4_K decode matvec (memory-bound decode path). Q4_K stores 256 weights in a 144-byte
// super-block: f16 d, f16 dmin, 12 packed 6-bit scale bytes, then 128 quant bytes (two 4-bit
// weights each). At ~4.5 bits/weight this reads ~7x fewer bytes than dense f16 -> ~7x the decode
// bandwidth on this ~217 GB/s APU. ASYMMETRIC: weight = d*sc*q - dmin*m, where (sc,m) come from
// get_scale_min_k4(j) over the 12 scale bytes; 8 sub-blocks of 32. The decode MUST be bit-faithful
// to the CPU oracle k_quants::BlockQ4K::to_float (and the Vulkan mul_mat_vec_q4k.comp port).
//
// One WARP per output row (mirrors qmatvec_q8_0_*), but -- unlike Q8_0 (32-wide blocks, so a row
// has >=128 blocks at k>=4096 and every lane stays busy) -- a Q4_K super-block is 256-wide, so a
// k=5120 row has only 20 super-blocks. Splitting whole super-blocks across lanes (the naive port)
// would leave 12/32 lanes idle AND give each active lane a 256-weight serial inner loop. Instead
// the whole WARP cooperates on each super-block: lane t (0..31) owns weight-POSITION t inside every
// 32-wide sub-block. Sub-block g (0..7) uses scale-pair g; its nibbles come from qs byte
// (g/2)*32 + t (LOW nibble for even g, HIGH for odd g -- exactly the to_float chunk layout: chunk
// c=g/2 covers qs[c*32, c*32+32), sub-block 2c = low nibbles, 2c+1 = high nibbles). Activation index
// is g*32 + t. So every lane is busy, the inner loop is 8 sub-blocks (not 256), and the per-lane
// partials reduce across the warp at the end.

#define Q4K_BLOCK_BYTES 144

// get_scale_min_k4 (k_quants/utils.rs): unpack the 6-bit scale `sc` and min `m` for sub-block j
// (0..7) from the 12 packed scale bytes `s`. Returns sc in [0..63], m in [0..63].
__device__ __forceinline__ void q4k_scale_min(const uint8_t* __restrict__ s, int j, int* sc, int* m) {
    if (j < 4) {
        *sc = s[j] & 63;
        *m  = s[j + 4] & 63;
    } else {
        *sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
        *m  = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
}

// Lane `lane` (0..31)'s partial dot over one super-block: sums its weight-position across all 8
// sub-blocks. Templated on activation type so f16/bf16 share the exact f32 math. `blk` -> 144-byte
// block, `xb` -> the 256 activations for this super-block.
//
// Iterate the 4 chunks (not 8 sub-blocks): chunk c owns qs byte (c*32 + lane), which packs BOTH
// sub-block 2c (low nibble, activation (2c)*32+lane) and 2c+1 (high nibble, activation (2c+1)*32+lane).
// Reading the byte ONCE per chunk halves the qs DRAM traffic vs reading it per sub-block, and the
// per-lane qs read is contiguous across the warp (lanes 0..31 -> bytes c*32..c*32+31), so it
// coalesces. The 16-byte header (d/dmin/12 scales) is a warp-uniform broadcast load (served from
// cache), and the scale unpack is cheap ALU.
template <typename XT>
__device__ __forceinline__ float q4k_lane_partial(const uint8_t* __restrict__ blk, const XT* __restrict__ xb, int lane) {
    // blk is 2-byte aligned (144 is a multiple of 2, base is device-aligned), so the f16 reads align.
    const float d    = __half2float(*reinterpret_cast<const __half*>(blk));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(blk + 2));
    const uint8_t* scales = blk + 4;        // 12 packed scale bytes
    const uint8_t* qs     = blk + 16;       // 128 quant bytes
    float s = 0.0f;
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        int sc_lo, m_lo, sc_hi, m_hi;
        q4k_scale_min(scales, 2 * c,     &sc_lo, &m_lo);
        q4k_scale_min(scales, 2 * c + 1, &sc_hi, &m_hi);
        const int qb = qs[c * 32 + lane];   // one read, both nibbles
        const float wlo = d * (float)sc_lo * (float)(qb & 0xF) - dmin * (float)m_lo;
        const float whi = d * (float)sc_hi * (float)(qb >> 4)  - dmin * (float)m_hi;
        s += wlo * (float)xb[(2 * c) * 32 + lane];
        s += whi * (float)xb[(2 * c + 1) * 32 + lane];
    }
    return s;
}

extern "C" __global__ void qmatvec_q4k_f16(
    const int nrows,
    const int ncols,            // multiple of 256
    const uint8_t* __restrict__ wq,
    const __half* __restrict__ x,
    __half* __restrict__ y
) {
    const int lane = threadIdx.x & 31;
    const int rows_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * rows_per_block + (threadIdx.x >> 5);
    if (row >= nrows) {
        return;
    }
    const int nblocks = ncols >> 8; // ncols / 256
    const uint8_t* row_ptr = wq + (size_t)row * (size_t)nblocks * Q4K_BLOCK_BYTES;

    // Whole warp cooperates on each super-block; lane owns weight-position `lane` in every sub-block.
    float acc = 0.0f;
    for (int b = 0; b < nblocks; ++b) {
        const uint8_t* blk = row_ptr + (size_t)b * Q4K_BLOCK_BYTES;
        const __half* xb = x + (size_t)b * 256;
        acc += q4k_lane_partial<__half>(blk, xb, lane);
    }

    // Warp reduction (wave32).
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off);
    }
    if (lane == 0) {
        y[row] = __float2half(acc);
    }
}

// BF16-native Q4_K decode matvec: identical f32 math to qmatvec_q4k_f16, reading/writing bf16 so the
// decode path keeps the model's working dtype end-to-end (no bf16->f32->f16 cast detour).
extern "C" __global__ void qmatvec_q4k_bf16(
    const int nrows,
    const int ncols,            // multiple of 256
    const uint8_t* __restrict__ wq,
    const hip_bfloat16* __restrict__ x,
    hip_bfloat16* __restrict__ y
) {
    const int lane = threadIdx.x & 31;
    const int rows_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * rows_per_block + (threadIdx.x >> 5);
    if (row >= nrows) {
        return;
    }
    const int nblocks = ncols >> 8; // ncols / 256
    const uint8_t* row_ptr = wq + (size_t)row * (size_t)nblocks * Q4K_BLOCK_BYTES;

    float acc = 0.0f;
    for (int b = 0; b < nblocks; ++b) {
        const uint8_t* blk = row_ptr + (size_t)b * Q4K_BLOCK_BYTES;
        const hip_bfloat16* xb = x + (size_t)b * 256;
        acc += q4k_lane_partial<hip_bfloat16>(blk, xb, lane);
    }

    // Warp reduction (wave32).
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off);
    }
    if (lane == 0) {
        y[row] = hip_bfloat16(acc);
    }
}

// ==============================================================================================
// UNIFIED quant DECODE matvec core (Cut 1, decode side). ONE quant-agnostic warp-per-row core
// (`qmatvec_core<WTYPE,XT>`) + ONE per-type device decode (`qdec<WTYPE>::partial<XT>`) + ONE
// `qdw_traits<WTYPE>` row covers the whole 1-bit -> 8-bit zoo. NO per-quant kernel: adding a type
// is one decode struct + one traits row + one launcher table entry. This mirrors the CPU
// `quant_format!` / `for_each_quant!` decomplection on the GPU.
//
// Scientist Cut-1 invariant: every GGML quant decodes to (int quant) * (per-block f32 scale)
// [+ optional min], so the WHOLE accumulation needs only TWO shapes, both expressed by the same
// per-lane partial-dot contract:
//   SYMMETRIC  : val(pos) = scale * q(pos)                 (Q8_0/Q4_0/Q6_K/IQ4_XS-via-LUT/TQ2_0)
//   ASYMMETRIC : val(pos) = d * sc * q(pos) - dmin * m     (Q4_K, and IQ1 via a +delta bias)
// Codebook types (IQ4_XS/IQ4_NL/MXFP4/NVFP4) ride the symmetric shape through a kvalues int8 LUT.
//
// WARP STRATEGY (unifies Q8_0's 32-wide blocks and Q4_K's 256-wide super-blocks): the WHOLE warp
// cooperates on ONE block at a time; lane L owns the ELEMS/32 element POSITIONS { e*32 + L } for
// e in [0, ELEMS/32). For a 32-elem block (Q8_0/Q4_0) that is exactly 1 position/lane; for a
// 256-elem super-block (Q4_K/Q6_K/IQ4_XS/TQ2_0) it is 8. This is the generalization of the proven
// `q4k_lane_partial` to any block size: the per-type decode returns lane L's partial dot over its
// owned positions, the core warp-reduces. Activation index for position p is just `xb[p]` (the
// per-type position->bits mapping that mirrors `to_float` lives entirely inside the decode). The
// original `qmatvec_q8_0_*` / `qmatvec_q4k_*` kernels above are kept verbatim as the proven
// references; the launcher routes through the unified `qmatvecu_*` entry points at the bottom.

#ifdef __HIPCC__
// int8 codebook LUT for the NL/XS family (KVALUES_IQ4NL, k_quants.rs:43). MXFP4/NVFP4 would add
// their own LUT the same way -- the decode just indexes a 16-entry table to an int8 quant.
__device__ __constant__ int8_t KVALUES_IQ4NL_D[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};
#endif

// Decode weight types (decode side -- a distinct id space from the prefill WT_* above, which is
// #undef'd at end-of-file). One row per wired type; BYTES = on-disk block stride, ELEMS = weights
// per block, SYMMETRIC = which of the two accumulation shapes (documentation/dispatch; the
// partial-dot contract folds min/delta inside the decode so the core itself is shape-agnostic).
#define DW_Q8_0   0   // 34 B,  32 elems, symmetric 8-bit.            val = d*qs[p]
#define DW_Q4_0   1   // 18 B,  32 elems, symmetric nibble (-8).      val = d*((nib)-8)
#define DW_Q4_K   2   // 144 B, 256 elems, ASYMMETRIC super-block.    val = d*sc*q - dmin*m
#define DW_Q6_K   3   // 210 B, 256 elems, symmetric K-quant (6-bit). val = d*sc*(q-32)
#define DW_IQ4_XS 4   // 136 B, 256 elems, symmetric codebook (LUT).  val = d*(ls-32)*KVALUES[idx]
#define DW_TQ2_0  5   // 66 B,  256 elems, symmetric ternary (2-bit). val = d*((q&3)-1)

template <int WTYPE> struct qdw_traits;
template <> struct qdw_traits<DW_Q8_0>   { static constexpr int BYTES = 34;  static constexpr int ELEMS = 32;  static constexpr bool SYMMETRIC = true;  static constexpr int NSC = 1; };
template <> struct qdw_traits<DW_Q4_0>   { static constexpr int BYTES = 18;  static constexpr int ELEMS = 32;  static constexpr bool SYMMETRIC = true;  static constexpr int NSC = 1; };
template <> struct qdw_traits<DW_Q4_K>   { static constexpr int BYTES = 144; static constexpr int ELEMS = 256; static constexpr bool SYMMETRIC = false; static constexpr int NSC = 8; };
template <> struct qdw_traits<DW_Q6_K>   { static constexpr int BYTES = 210; static constexpr int ELEMS = 256; static constexpr bool SYMMETRIC = true;  static constexpr int NSC = 16; };
template <> struct qdw_traits<DW_IQ4_XS> { static constexpr int BYTES = 136; static constexpr int ELEMS = 256; static constexpr bool SYMMETRIC = true;  static constexpr int NSC = 8; };
template <> struct qdw_traits<DW_TQ2_0>  { static constexpr int BYTES = 66;  static constexpr int ELEMS = 256; static constexpr bool SYMMETRIC = true;  static constexpr int NSC = 1; };

// Per-type decode. `partial<XT>(blk, xb, lane)` returns lane L's partial dot over its owned
// positions { e*32 + L } of ONE block, mirroring the CPU `to_float` for that type bit-for-bit.
// Function templates cannot partial-specialize, so each type is a struct with a templated method.
template <int WTYPE> struct qdec;

// Type-dispatched f32 -> output store (the only place the element type leaks): __half needs
// __float2half, hip_bfloat16 takes float in its ctor. Keeps the core's epilogue type-agnostic.
__device__ __forceinline__ void qstore(__half* p, float v)        { *p = __float2half(v); }
__device__ __forceinline__ void qstore(hip_bfloat16* p, float v)  { *p = hip_bfloat16(v); }

// Q8_0: f16 d at byte 0, 32 int8 quants at byte +2. pos p (= lane) -> d * qs[p]. (BlockQ8_0::to_float)
template <> struct qdec<DW_Q8_0> {
    template <typename XT>
    static __device__ __forceinline__ float partial(const uint8_t* __restrict__ blk, const XT* __restrict__ xb, int lane) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
        return d * (float)qs[lane] * (float)xb[lane];
    }
};

// Q4_0: f16 d at byte 0, 16 nibble-pairs at byte +2. pos p (= lane): p<16 -> low nibble of qs[p];
// p>=16 -> high nibble of qs[p-16]; val = d*(nibble-8). (BlockQ4_0::to_float, k_quants.rs:235.)
template <> struct qdec<DW_Q4_0> {
    template <typename XT>
    static __device__ __forceinline__ float partial(const uint8_t* __restrict__ blk, const XT* __restrict__ xb, int lane) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint8_t* qs = blk + 2;
        const int nib = (lane < 16) ? (qs[lane] & 0x0F) : (qs[lane - 16] >> 4);
        return d * (float)(nib - 8) * (float)xb[lane];
    }
};

// Q4_K: ASYMMETRIC. Reuses the validated q4k_lane_partial (8 sub-blocks via the 4-chunk loop; lane
// owns position `lane` in every sub-block = positions { e*32 + lane }). val = d*sc*q - dmin*m.
template <> struct qdec<DW_Q4_K> {
    template <typename XT>
    static __device__ __forceinline__ float partial(const uint8_t* __restrict__ blk, const XT* __restrict__ xb, int lane) {
        return q4k_lane_partial<XT>(blk, xb, lane);
    }
};

// Q6_K: symmetric 6-bit. Block = ql[128], qh[64], scales[16] (SIGNED int8), d (f16) = 210 B.
// Two 128-element halves; within a half the to_float (k_quants.rs:2398) packs 4 quadrants. Lane L
// owns positions { e*32 + L }, e in 0..7; for each e: half=e/4, quadrant=e%4, ll=L, is=L/16.
template <> struct qdec<DW_Q6_K> {
    template <typename XT>
    static __device__ __forceinline__ float partial(const uint8_t* __restrict__ blk, const XT* __restrict__ xb, int lane) {
        const uint8_t* ql_b = blk;            // 128
        const uint8_t* qh_b = blk + 128;      // 64
        const int8_t*  sc_b = reinterpret_cast<const int8_t*>(blk + 192); // 16 signed
        const float d = __half2float(*reinterpret_cast<const __half*>(blk + 208));
        const int is = lane >> 4;             // l/16 in [0,1]
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int half = e >> 2;          // 0,1
            const int quad = e & 3;           // 0..3
            const uint8_t* ql = ql_b + 64 * half;
            const uint8_t* qh = qh_b + 32 * half;
            const int8_t*  sc = sc_b + 8 * half;
            int q; int sci;
            if (quad == 0)      { q = ((ql[lane]      & 0xF) | ((qh[lane] & 3) << 4)) - 32; sci = is;     }
            else if (quad == 1) { q = ((ql[lane + 32] & 0xF) | (((qh[lane] >> 2) & 3) << 4)) - 32; sci = is + 2; }
            else if (quad == 2) { q = ((ql[lane]      >> 4)  | (((qh[lane] >> 4) & 3) << 4)) - 32; sci = is + 4; }
            else                { q = ((ql[lane + 32] >> 4)  | (((qh[lane] >> 6) & 3) << 4)) - 32; sci = is + 6; }
            const float val = d * (float)sc[sci] * (float)q;
            s += val * (float)xb[e * 32 + lane];
        }
        return s;
    }
};

// IQ4_XS: symmetric codebook. Block = d (f16), scales_h (u16), scales_l[4], qs[128] = 136 B.
// 8 sub-blocks of 32; 6-bit scale ls; dl = d*(ls-32); val = dl*KVALUES_IQ4NL[idx]. Lane L owns
// positions { e*32 + L }, so sub-block ib=e, jj=L: L<16 -> low nibble qs[e*16+L], else high nibble
// qs[e*16+(L-16)]. (BlockIQ4xs::to_float, k_quants.rs:818.)
template <> struct qdec<DW_IQ4_XS> {
    template <typename XT>
    static __device__ __forceinline__ float partial(const uint8_t* __restrict__ blk, const XT* __restrict__ xb, int lane) {
        const float d_all = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint16_t scales_h = *reinterpret_cast<const uint16_t*>(blk + 2);
        const uint8_t* scales_l = blk + 4;    // 4 bytes
        const uint8_t* qs = blk + 8;          // 128 bytes
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int ib = e;
            const int ls = ((scales_l[ib >> 1] >> (4 * (ib & 1))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
            const float dl = d_all * (float)(ls - 32);
            int idx;
            if (lane < 16) idx = qs[ib * 16 + lane] & 0x0F;
            else           idx = qs[ib * 16 + (lane - 16)] >> 4;
            const float val = dl * (float)KVALUES_IQ4NL_D[idx];
            s += val * (float)xb[e * 32 + lane];
        }
        return s;
    }
};

// TQ2_0: symmetric ternary (2-bit). Block = qs[64], d (f16) = 66 B. to_float (iq_quants.rs:102)
// fills via for j(step 32){ for l(0..4){ for m(0..32){ } } }: global pos p -> half=p/128, l=(p%128)/32,
// m=p%32, byte=half*32+m, val=((qs[byte]>>(l*2))&3 - 1)*d. Lane L owns { e*32+L }: half=e/4, l=e%4, m=L.
template <> struct qdec<DW_TQ2_0> {
    template <typename XT>
    static __device__ __forceinline__ float partial(const uint8_t* __restrict__ blk, const XT* __restrict__ xb, int lane) {
        const uint8_t* qs = blk;              // 64
        const float d = __half2float(*reinterpret_cast<const __half*>(blk + 64));
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int half = e >> 2;          // 0,1
            const int l = e & 3;              // 0..3
            const int byte = half * 32 + lane;
            const int q = (qs[byte] >> (l * 2)) & 3;
            const float val = (float)(q - 1) * d;
            s += val * (float)xb[e * 32 + lane];
        }
        return s;
    }
};

// Whole-block decode: lane L computes the COMPLETE dot of ONE block over ALL its ELEMS positions,
// reusing the per-type `qdec<WTYPE>::partial` per-position math. Because `partial(blk,xb,p)` returns
// the dot over the position-set { e*32 + p } owned by "lane" p, summing it over p = 0..31 covers
// every position of the block exactly once -> bit-faithful to the CPU `to_float` (math identical;
// only the f32 accumulation ORDER changes, which the 1%-of-magnitude numeric gate explicitly allows).
template <int WTYPE, typename XT>
__device__ __forceinline__ float qdec_block_full(const uint8_t* __restrict__ blk, const XT* __restrict__ xb) {
    float s = 0.0f;
    #pragma unroll
    for (int p = 0; p < 32; ++p) {
        s += qdec<WTYPE>::template partial<XT>(blk, xb, p);
    }
    return s;
}

// THE unified decode matvec core. Warp per output row. LANE-STRIDED over blocks: lane L streams
// whole blocks b = L, L+32, L+64, ... so the 32 lanes issue 32 INDEPENDENT block loads in flight at
// once, hiding LPDDR5X latency on the streaming m=1 FFN shapes (the old whole-warp-serial-per-block
// loop re-read the scale every block and could only keep one block's loads in flight -> 50-61% of
// read-peak). Each lane fully decodes its blocks via `qdec_block_full`; a final warp-shuffle reduces
// the per-lane partials. Bit-faithful for every WTYPE (Q8_0/Q4_0/Q4_K/Q6_K/IQ4_XS/TQ2_0): only the
// block-iteration pattern changed, the per-type decode + warp reduce are preserved.
template <int WTYPE, typename XT>
__device__ __forceinline__ void qmatvec_core(
    const int nrows,
    const int ncols,                  // multiple of ELEMS
    const uint8_t* __restrict__ wq,
    const XT* __restrict__ x,
    XT* __restrict__ y
) {
    constexpr int WBYTES = qdw_traits<WTYPE>::BYTES;
    constexpr int ELEMS  = qdw_traits<WTYPE>::ELEMS;
    const int lane = threadIdx.x & 31;
    const int rows_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * rows_per_block + (threadIdx.x >> 5);
    if (row >= nrows) {
        return;
    }
    const int nblocks = ncols / ELEMS;
    const uint8_t* row_ptr = wq + (size_t)row * (size_t)nblocks * WBYTES;

    float acc = 0.0f;
    for (int b = lane; b < nblocks; b += 32) {
        acc += qdec_block_full<WTYPE, XT>(row_ptr + (size_t)b * WBYTES, x + (size_t)b * ELEMS);
    }
    // Warp reduction (wave32).
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off);
    }
    if (lane == 0) {
        qstore(&y[row], acc);
    }
}

// Unified per-type entry points (f16 + bf16). These REPLACE the per-quant launcher dispatch: the
// Rust side selects the symbol by GgmlDType, every symbol is the SAME core with a different WTYPE.
#define DEFINE_QMATVECU(NAME, WTYPE)                                                                 \
    extern "C" __global__ void qmatvecu_##NAME##_f16(                                                \
        const int nrows, const int ncols, const uint8_t* __restrict__ wq,                            \
        const __half* __restrict__ x, __half* __restrict__ y) {                                      \
        qmatvec_core<WTYPE, __half>(nrows, ncols, wq, x, y);                                          \
    }                                                                                                \
    extern "C" __global__ void qmatvecu_##NAME##_bf16(                                               \
        const int nrows, const int ncols, const uint8_t* __restrict__ wq,                            \
        const hip_bfloat16* __restrict__ x, hip_bfloat16* __restrict__ y) {                          \
        qmatvec_core<WTYPE, hip_bfloat16>(nrows, ncols, wq, x, y);                                    \
    }

DEFINE_QMATVECU(q8_0,  DW_Q8_0)
DEFINE_QMATVECU(q4_0,  DW_Q4_0)
DEFINE_QMATVECU(q4k,   DW_Q4_K)
DEFINE_QMATVECU(q6k,   DW_Q6_K)
DEFINE_QMATVECU(iq4xs, DW_IQ4_XS)
DEFINE_QMATVECU(tq2_0, DW_TQ2_0)

// THE unified indexed-MoE decode core. Same warp-per-row + lane-strided-block math as
// `qmatvec_core`, but ALL routed slots run in ONE launch with the routed expert on grid.y:
//   slot s = blockIdx.y, expert = ids[s] (read ON-DEVICE -- no host ids round-trip), and the warp
//   computes output row `row` of THAT expert's [n,k] byte-slice of the resident `[E,n,k]` bank
//   against slot s's activation row (x + s*ncols) into y[s*n + row].
// This is the non-Q4_K twin of `moe_qmatvec_q4k_dp4a_*`: it collapses the per-expert host launch
// loop (which had to materialize ids on the host -> hipErrorStreamCaptureImplicit, breaking HIP
// graph capture) into one well-occupied capture-clean grid, while reusing the proven per-type
// `qdec_block_full<WTYPE,XT>` decode verbatim -- only the expert-offset + slot indexing change.
template <int WTYPE, typename XT>
__device__ __forceinline__ void moe_qmatvec_core(
    const int n,                       // output rows per expert (weight rows)
    const int ncols,                   // k, multiple of ELEMS
    const int nslots,                  // routed slots (= nrows)
    const uint8_t* __restrict__ wbank, // [E, n, k] resident GGML blocks
    const int* __restrict__ ids,       // [nslots] expert id per slot
    const XT* __restrict__ x,          // [nslots, ncols] routed activations
    XT* __restrict__ y                 // [nslots, n]
) {
    constexpr int WBYTES = qdw_traits<WTYPE>::BYTES;
    constexpr int ELEMS  = qdw_traits<WTYPE>::ELEMS;
    const int s = blockIdx.y;
    if (s >= nslots) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const int rows_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * rows_per_block + (threadIdx.x >> 5);
    if (row >= n) {
        return;
    }
    const int nblocks = ncols / ELEMS;
    const int expert = ids[s];
    // Offset the bank to expert `ids[s]`, row `row`, IN-KERNEL (the host loop used to do this).
    const uint8_t* row_ptr =
        wbank + ((size_t)expert * (size_t)n + (size_t)row) * (size_t)nblocks * WBYTES;
    const XT* x_row = x + (size_t)s * (size_t)ncols;

    float acc = 0.0f;
    for (int b = lane; b < nblocks; b += 32) {
        acc += qdec_block_full<WTYPE, XT>(row_ptr + (size_t)b * WBYTES, x_row + (size_t)b * ELEMS);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off);
    }
    if (lane == 0) {
        qstore(&y[(size_t)s * (size_t)n + (size_t)row], acc);
    }
}

// Unified per-type indexed-MoE entry points (f16 + bf16). Twin of DEFINE_QMATVECU: every symbol is
// the SAME `moe_qmatvec_core` with a different WTYPE -- one core, one launcher table on the Rust
// side. The Rust `moe_matvec_quant` selects the symbol by RocmQuantType + activation dtype.
#define DEFINE_MOE_QMATVECU(NAME, WTYPE)                                                             \
    extern "C" __global__ void moe_qmatvecu_##NAME##_f16(                                            \
        const int n, const int ncols, const int nslots, const uint8_t* __restrict__ wbank,          \
        const int* __restrict__ ids, const __half* __restrict__ x, __half* __restrict__ y) {        \
        moe_qmatvec_core<WTYPE, __half>(n, ncols, nslots, wbank, ids, x, y);                         \
    }                                                                                                \
    extern "C" __global__ void moe_qmatvecu_##NAME##_bf16(                                           \
        const int n, const int ncols, const int nslots, const uint8_t* __restrict__ wbank,          \
        const int* __restrict__ ids, const hip_bfloat16* __restrict__ x, hip_bfloat16* __restrict__ y) { \
        moe_qmatvec_core<WTYPE, hip_bfloat16>(n, ncols, nslots, wbank, ids, x, y);                   \
    }

DEFINE_MOE_QMATVECU(q8_0,  DW_Q8_0)
DEFINE_MOE_QMATVECU(q4_0,  DW_Q4_0)
DEFINE_MOE_QMATVECU(q4k,   DW_Q4_K)
DEFINE_MOE_QMATVECU(q6k,   DW_Q6_K)
DEFINE_MOE_QMATVECU(iq4xs, DW_IQ4_XS)
DEFINE_MOE_QMATVECU(tq2_0, DW_TQ2_0)

#undef DEFINE_MOE_QMATVECU
#undef DEFINE_QMATVECU
#undef DW_Q8_0
#undef DW_Q4_0
#undef DW_Q4_K
#undef DW_Q6_K
#undef DW_IQ4_XS
#undef DW_TQ2_0

// ----------------------------------------------------------------------------------------------
// Native Q8_0 quant GEMM (prefill path) using RDNA3 WMMA (matrix cores).
//   Y[M,N] = X[M,K] (f16) * W[N,K]^T  with W stored Q8_0 [N,K].
// One wave (32 lanes) per 16x16 output tile; tiles are laid out 1-D over the grid (row-major in
// (row_tile, col_tile)). Each K-step (16) stages a 16x16 X tile and a dequantized 16x16 W tile to
// shared, then one wmma 16x16x16 f16->f32 MAC. Keeping W in Q8_0 means no resident dense f16 copy
// (which would slow decode) and the MAC runs on the matrix cores instead of rocBLAS.
#if defined(__HIPCC__) && (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__))
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));

// 64x64 block tile, 4 waves (128 threads), each wave a 32x32 register tile (2x2 WMMA frags = 4
// f32 accumulators). The 4 waves share double-buffered shared tiles: each iteration prefetches the
// NEXT K-tile's X + dequantized W into the other buffer while the matrix cores run the current
// tile's MACs, hiding the global-load + dequant latency. BK=16; ncol_tiles = ceil(N/64).
#define BK 16
#define STG(BUF, K0)                                                                       \
    do {                                                                                   \
        _Pragma("unroll")                                                                  \
        for (int j = 0; j < 8; ++j) {                                                       \
            const int idx = t + j * 128; /* 0..1023 */                                      \
            const int m = idx >> 4;       /* 0..63   */                                      \
            const int kk = idx & 15;                                                        \
            const int gr = row_tile + m;                                                    \
            sX[(BUF)][idx] =                                                                 \
                (gr < M) ? (_Float16)__half2float(x[(size_t)gr * K + ((K0) + kk)]) : (_Float16)0; \
            const int gn = col_tile + m;                                                    \
            if (gn < N) {                                                                   \
                const int gk = (K0) + kk;                                                    \
                const uint8_t* blk = wq + ((size_t)gn * nblocks + (gk >> 5)) * 34;           \
                const float d = __half2float(*reinterpret_cast<const __half*>(blk));         \
                const int q = (int)(reinterpret_cast<const int8_t*>(blk + 2)[gk & 31]);      \
                sW[(BUF)][idx] = (_Float16)(d * (float)q);                                   \
            } else {                                                                        \
                sW[(BUF)][idx] = (_Float16)0;                                                \
            }                                                                               \
        }                                                                                   \
    } while (0)

extern "C" __global__ void qgemm_q8_0_f16(
    const int M,
    const int N,
    const int K,            // multiple of 32
    const int ncol_tiles,   // ceil(N/64)
    const __half* __restrict__ x,    // [M, K]
    const uint8_t* __restrict__ wq,  // [N, K] Q8_0
    __half* __restrict__ y           // [M, N]
) {
    const int tile = blockIdx.x;
    const int row_tile = (tile / ncol_tiles) * 64;
    const int col_tile = (tile % ncol_tiles) * 64;
    const int t = threadIdx.x;       // 0..127
    const int lane = t & 31;
    const int wave_m = (t >> 5) >> 1; // 0,1  (row sub-block * 32)
    const int wave_n = (t >> 5) & 1;  // 0,1  (col sub-block * 32)
    const int nblocks = K >> 5;
    const int numK = K >> 4;          // K / 16

    __shared__ _Float16 sX[2][64 * 16];
    __shared__ _Float16 sW[2][64 * 16];

    v8f acc00 = {0,0,0,0,0,0,0,0};
    v8f acc01 = {0,0,0,0,0,0,0,0};
    v8f acc10 = {0,0,0,0,0,0,0,0};
    v8f acc11 = {0,0,0,0,0,0,0,0};

    STG(0, 0);
    __syncthreads();

    for (int i = 0; i < numK; ++i) {
        const int cur = i & 1;
        if (i + 1 < numK) {
            STG(cur ^ 1, (i + 1) << 4); // prefetch next K-tile while the MACs below run
        }
        // Each wave loads its 2 M-fragments and 2 N-fragments (lane l = tile row l%16), then 4 MACs.
        const int aoff = (wave_m * 32 + (lane & 15)) * 16;
        const int boff = (wave_n * 32 + (lane & 15)) * 16;
        v16h a0, a1, b0, b1;
        #pragma unroll
        for (int e = 0; e < 16; ++e) {
            a0[e] = sX[cur][aoff + e];
            a1[e] = sX[cur][aoff + 16 * 16 + e];
            b0[e] = sW[cur][boff + e];
            b1[e] = sW[cur][boff + 16 * 16 + e];
        }
        acc00 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc00);
        acc01 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b1, acc01);
        acc10 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b0, acc10);
        acc11 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc11);
        __syncthreads();
    }

    // Store. RDNA3 wave32 f32 layout: lane l holds column (l%16) of a 16x16 sub-tile; element e is
    // row (e*2 + l/16).
    const int lcol = lane & 15;
    const int lrow = lane >> 4;
    const int rbase = row_tile + wave_m * 32;
    const int cbase = col_tile + wave_n * 32;
    const int c0 = cbase + lcol;
    const int c1 = cbase + 16 + lcol;
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int r0 = rbase + e * 2 + lrow;
        const int r1 = rbase + 16 + e * 2 + lrow;
        if (c0 < N) {
            if (r0 < M) y[(size_t)r0 * N + c0] = __float2half(acc00[e]);
            if (r1 < M) y[(size_t)r1 * N + c0] = __float2half(acc10[e]);
        }
        if (c1 < N) {
            if (r0 < M) y[(size_t)r0 * N + c1] = __float2half(acc01[e]);
            if (r1 < M) y[(size_t)r1 * N + c1] = __float2half(acc11[e]);
        }
    }
}

// Native Q8_0 × int8 GEMM (prefill) on RDNA3 int8 matrix cores -- llama's mul_mat_q path.
//   Y[M,N] = sum_blocks (xd[m,blk] * wd[n,blk]) * sum_{k in 32-block} xq[m,k] * wq_i8[n,k]
// Weights STAY int8 (no f16 dequant -> half the shared traffic), int8 matrix cores (~2x f16).
//
// CONFIG NOTE (vs llama RDNA3 mmq). llama (mmq.cuh, AMD_WMMA_AVAILABLE): 128x128 tile, MMQ_NWARPS=8
// (256 threads), granularity=32 -> ntx=2 -> 8 frags/warp, and -- critically -- it PRE-QUANTIZES the
// activations ONCE into a separate int8 q8_1 buffer (quantize_mmq_q8_1) before the GEMM. Round 1
// FUSED the activation quant inside this kernel and RE-QUANTIZED every activation row once per
// N-col-tile (~96x for a 12288-wide FFN), which both wasted bandwidth/absmax-reductions AND pushed
// the 8-warp/8-frag layout to 190 VGPRs (occupancy 8 waves/SIMD32), so we fell back to 16 warps.
//
// Round 2 DE-FUSES exactly like llama: RocmDevice::qmmq_q8_0 now calls quantize_q8 first (the
// quantize_q8 kernel emits int8 xq + per-32-block f16 xd in VRAM, byte-identical to the old fused
// path), and this kernel consumes the pre-quantized xq/xd. The X staging is now a plain strided
// int8 copy + a per-row scale copy (same shape as the W staging) -- no in-kernel absmax shuffle, no
// per-tile re-quant. That removes the redundant activation traffic and drops VGPRs (no in-kernel
// reduce). We keep llama's 128x128 tile + int8 path + double-buffer; weight/activation scales are
// staged to shared once per block (like llama's x_df) and fragments load 16-byte vectorized (like
// ggml_cuda_memcpy_1<16>).
//
// DESIGN-SPACE MAP (measured on gfx1151/ROCm7.x, Qwen3-8B-Q8_0, 512-tok prefill):
//   - THIS kernel (double-buffer, 1-block K, 16-warp/4-frag, VECTORIZED 16B staging, 124 VGPR,
//     10 waves, 0 spill):                                                                          627 t/s
//   - prior baseline (same, but byte-wise staging, 150 VGPR, 9 waves):                             467 t/s
//   - single-buffer narrow / WIDE-K=2 / WIDE-K=4 / double-buffer WIDE-K (all prior round):     315-452 t/s
//   - 8-warp/8-frag (resident-accumulator decomp; 192-256 VGPR + spills at narrow K):           84-425 t/s
//
// GROUND TRUTH vs llama (disassembled C:\llama\hip\ggml-hip.dll, gfx1151 code object, mmq Q8_0
// kernel-descriptor .vgpr_count / metadata): llama's RDNA3 mmq is NOT register-lean and does NOT run
// at high occupancy -- its working tiles are 168 VGPR (mmq_x=64) .. 240 VGPR (mmq_x=128), 0 spills,
// 6-9 waves/SIMD, 256 threads (MMQ_NWARPS=8). So registers/occupancy are NOT what separate us from
// llama's 1170 t/s (we already sit at 124 VGPR / 10 waves / 0 spills -- LEANER than llama). The two
// levers that ARE portable and that we use here:
//   (a) llama's ggml_cuda_memcpy_1<16> 16-byte coalesced staging -> the +34% jump above. Our X (int8
//       activations) staging now issues global_load_b128 (1 16B load/thread) instead of byte loads.
//   (b) transient int32 WMMA accumulator folded into a small f32 sum (already done here: ic** live
//       only inside the block loop; acc** f32 are the only resident accumulators).
// REMAINING GAP to 1170: W staging cannot coalesce past 2-byte loads because raw GGML Q8_0 interleaves
// a 2-byte f16 scale every 34 bytes (quants land at byte 34*blk+2, only 2B-aligned). llama side-steps
// this by REPACKING weights into separate contiguous quant/scale arrays (block_q8_1_mmq) so both load
// fully coalesced, AND by stream-K work partitioning that keeps all CUs busy. Those are the next levers
// (a one-time weight repack + stream-K decomposition), not a register/fragment change.
//
// 16 warps (512 threads) in a 4x4 grid of 32x32 sub-tiles over the 128x128 tile; each warp owns a
// 32x32 region = 2x2 = 4 WMMA fragments (4 f32 accumulators). Double-buffered int8 shared prefetch;
// per 32-K block: 2 iu8 WMMAs/frag -> int32, then scale by per-block xd[m]*wd[n] into f32.
typedef int v4i_ld __attribute__((ext_vector_type(4)));

// ---------------------------------------------------------------------------------------------
// UNIFIED int8 quant GEMM core (Cut 1, prefill side). ONE quant-agnostic 128x128 / 16-warp / iu8-
// WMMA core (`qmmq_core<WTYPE>`) covers the SAME 1-bit -> 8-bit zoo the decode core does: the int8
// shared staging, the iu8 WMMA inner loop, and the int32->f32 scale epilogue are TYPE-INDEPENDENT.
// Only the per-block WEIGHT DECODE + the two accumulation SHAPES differ, both compile-time
// (`if constexpr` on `wt_traits<WTYPE>`), so the proven Q8_0/Q4_0 instantiations codegen exactly as
// before (every new branch elides for SYMMETRIC && SUBS==1 && SCALE_STEP==32). Adding a quant to
// prefill is ONE `decode_w_half<WTYPE>` (+ scale/min reads) + ONE `wt_traits<WTYPE>` row, NO new
// kernel -- the same invariant the decode core holds.
//
// Scientist Cut-1 invariant (prefill): the iu8 WMMA gives the int sum  iw = sum_k(q_w * q_x)  over
// each 32-element K-block (q_x = symmetric int8 activation, scale d_x = xd[row,blk]; q_w the per-
// type centered int8 weight). The WHOLE GEMM then needs only TWO accumulation shapes:
//   SYMMETRIC  : out += d_w * d_x * iw                         (Q8_0/Q4_0/Q6_K/IQ4_XS/TQ2_0)
//   ASYMMETRIC : out += d_w*sc * d_x * iw  -  dmin*m * d_x * ix (Q4_K/Q5_K; ix = sum_k q_x = the
//                q8_1 bias term, precomputed once per block by quantize_q8_1 into xs[row,blk])
// This is the prefill mirror of the decode core's per-element  d*sc*q - dmin*m: decode folds the
// min inside the per-element decode; prefill cannot (the int8 MAC is linear in q_w), so the min
// rides the separate ix bias term -- bit-faithful to the CPU q8_1 vec_dot (k_quants BlockQ4K).
//
// SUPER-BLOCKS. Q8_0/Q4_0 are 32-element blocks; the K-quants are 256-element super-blocks (8 sub-
// blocks of 32). The WMMA K-granularity is 32 (two iu8 k=16 MACs), so a 32-block index `blk` maps
// to super-block `blk/SUBS`, sub-block `g = blk%SUBS` (SUBS = SBLK_ELEMS/32). `decode_w_half` takes
// `g` so it indexes the right sub-block of the on-disk super-block (Q8_0/Q4_0: SUBS=1, g==0 -> the
// proven straight decode). SCALE GRANULARITY: most types carry one (sc,min) per 32-sub-block
// (SCALE_STEP=32); Q6_K carries a signed scale per 16 (SCALE_STEP=16) so its two WMMA halves scale
// independently (the `NSC16==2` path). The scale staging holds, per (col, 32-block): the main scale
// d_w*sc (slot sWd) and the secondary scalar sWm (= dmin*m for ASYM, = the 2nd-16 scale for Q6_K).
//
// Weight types (prefill id space; #undef'd at end-of-file, distinct from the decode DW_* ids):
#define WT_Q8_0   0   // 34 B,  32 elems, SYM 8-bit.            q_w = qs[k];        scale d
#define WT_Q4_0   1   // 18 B,  32 elems, SYM nibble.           q_w = nib-8;        scale d
#define WT_Q4_K   2   // 144 B, 256 elems, ASYM super-block.    q_w = nib(0..15);   scale d*sc, min dmin*m
#define WT_Q6_K   3   // 210 B, 256 elems, SYM K-quant 6-bit.   q_w = q-32;         scale d*sc (per-16)
#define WT_IQ4_XS 4   // 136 B, 256 elems, SYM codebook (LUT).  q_w = KVALUES[idx]; scale d*(ls-32)
#define WT_TQ2_0  5   // 66 B,  256 elems, SYM ternary 2-bit.   q_w = q-1;          scale d
// Per-type traits: BYTES = on-disk super-block stride; SBLK_ELEMS = elems/super-block (32 or 256);
// SYMMETRIC = no min bias; SCALE_STEP = elems per distinct weight scale (16 for Q6_K else 32).
// Derived: SUBS = 32-blocks per super-block; NSC16 = distinct scales per 32-block (2 iff SCALE_STEP=16).
template <int WTYPE> struct wt_traits;
template <> struct wt_traits<WT_Q8_0>   { static constexpr int BYTES = 34;  static constexpr int SBLK_ELEMS = 32;  static constexpr bool SYMMETRIC = true;  static constexpr int SCALE_STEP = 32; };
template <> struct wt_traits<WT_Q4_0>   { static constexpr int BYTES = 18;  static constexpr int SBLK_ELEMS = 32;  static constexpr bool SYMMETRIC = true;  static constexpr int SCALE_STEP = 32; };
template <> struct wt_traits<WT_Q4_K>   { static constexpr int BYTES = 144; static constexpr int SBLK_ELEMS = 256; static constexpr bool SYMMETRIC = false; static constexpr int SCALE_STEP = 32; };
template <> struct wt_traits<WT_Q6_K>   { static constexpr int BYTES = 210; static constexpr int SBLK_ELEMS = 256; static constexpr bool SYMMETRIC = true;  static constexpr int SCALE_STEP = 16; };
template <> struct wt_traits<WT_IQ4_XS> { static constexpr int BYTES = 136; static constexpr int SBLK_ELEMS = 256; static constexpr bool SYMMETRIC = true;  static constexpr int SCALE_STEP = 32; };
template <> struct wt_traits<WT_TQ2_0>  { static constexpr int BYTES = 66;  static constexpr int SBLK_ELEMS = 256; static constexpr bool SYMMETRIC = true;  static constexpr int SCALE_STEP = 32; };

// q4k_scale_min / KVALUES_IQ4NL_D are defined above for the decode core and reused verbatim here --
// ONE decode source per format (the prefill decode below mirrors the same bit layout as qdec<WTYPE>).

// Decode the 16-int8 HALF (k=half..half+15) of SUB-BLOCK `g` of one on-disk super-block `sb`,
// returned packed as a v4i_ld (16 bytes) so the caller's zero-init + single 16B shared store stays
// byte-identical to the validated Q8_0 staging. Element k maps to the SAME k the activation int8
// uses (k=0..31 sequential within the 32-block) so the iu8 MAC pairs them. `g` in [0,SUBS).
template <int WTYPE>
__device__ __forceinline__ v4i_ld decode_w_half(const uint8_t* sb, int g, int half);

// Q8_0: 32 int8 quants at byte +2 (SUBS=1, g==0). The 16B half is a straight load -- byte-identical
// to the pre-spread path (one memcpy of the 16B chunk at byte 34*blk+2+half).
template <>
__device__ __forceinline__ v4i_ld decode_w_half<WT_Q8_0>(const uint8_t* sb, int g, int half) {
    v4i_ld v;
    __builtin_memcpy(&v, sb + 2 + half, 16);
    return v;
}

// Q4_0: 16 nibble-pairs at byte +2 (SUBS=1, g==0). half=0 -> k 0..15 = LOW nibbles; half=16 ->
// k 16..31 = HIGH nibbles. q_w = nibble-8 (matches load_tiles_q4_0 / BlockQ4_0::to_float).
template <>
__device__ __forceinline__ v4i_ld decode_w_half<WT_Q4_0>(const uint8_t* sb, int g, int half) {
    const uint8_t* qs = sb + 2;
    const int shift = (half == 0) ? 0 : 4;
    int8_t tmp[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) tmp[i] = (int8_t)((int)((qs[i] >> shift) & 0x0F) - 8);
    v4i_ld v; __builtin_memcpy(&v, tmp, 16); return v;
}

// Q4_K: super-block = f16 d, f16 dmin, 12 scale bytes, 128 qs. Sub-block g (0..7) covers chunk
// c=g/2 of qs (byte c*32+k); even g = LOW nibble, odd g = HIGH nibble (the BlockQ4K::to_float / the
// q4k_lane_partial chunk layout). q_w = the UNSIGNED nibble 0..15 (the min term carries the offset).
// half=0 -> k 0..15, half=16 -> k 16..31 within the 32-element sub-block.
template <>
__device__ __forceinline__ v4i_ld decode_w_half<WT_Q4_K>(const uint8_t* sb, int g, int half) {
    const uint8_t* qs = sb + 16;
    const int c = g >> 1;            // chunk 0..3
    const int shift = (g & 1) ? 4 : 0;
    const uint8_t* qc = qs + c * 32 + half; // k=half..half+15 of this sub-block
    int8_t tmp[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) tmp[i] = (int8_t)((qc[i] >> shift) & 0x0F);
    v4i_ld v; __builtin_memcpy(&v, tmp, 16); return v;
}

// Q6_K: super-block = ql[128], qh[64], scales[16] (signed i8), d (f16). 32-sub-block g covers the
// SAME quadrant layout as qdec<DW_Q6_K>: half=g/4 of the block, quad=g%4. q_w = q-32 (signed,
// [-32,31]). k=half..half+15 within the sub-block (this routine returns one 16-half).
template <>
__device__ __forceinline__ v4i_ld decode_w_half<WT_Q6_K>(const uint8_t* sb, int g, int half) {
    const uint8_t* ql_b = sb;
    const uint8_t* qh_b = sb + 128;
    const int hf = g >> 2;           // 0,1  (which 128-half)
    const int quad = g & 3;          // 0..3
    const uint8_t* ql = ql_b + 64 * hf;
    const uint8_t* qh = qh_b + 32 * hf;
    int8_t tmp[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int kk = half + i;     // 0..31 within the sub-block (= ll in to_float)
        int q;
        if (quad == 0)      q = ((ql[kk]      & 0xF) | ((qh[kk] & 3) << 4)) - 32;
        else if (quad == 1) q = ((ql[kk + 32] & 0xF) | (((qh[kk] >> 2) & 3) << 4)) - 32;
        else if (quad == 2) q = ((ql[kk]      >> 4)  | (((qh[kk] >> 4) & 3) << 4)) - 32;
        else                q = ((ql[kk + 32] >> 4)  | (((qh[kk] >> 6) & 3) << 4)) - 32;
        tmp[i] = (int8_t)q;
    }
    v4i_ld v; __builtin_memcpy(&v, tmp, 16); return v;
}

// IQ4_XS: super-block = d (f16), scales_h (u16), scales_l[4], qs[128]. Sub-block ib=g covers
// qs[g*16 .. g*16+16]; low nibble -> k 0..15, high -> k 16..31. q_w = KVALUES_IQ4NL[idx] (signed
// int8 codebook). The per-sub-block scale d*(ls-32) rides decode_w_mainscale below.
template <>
__device__ __forceinline__ v4i_ld decode_w_half<WT_IQ4_XS>(const uint8_t* sb, int g, int half) {
    const uint8_t* qs = sb + 8 + g * 16;   // 16 packed-nibble bytes for sub-block g
    int8_t tmp[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int idx = (half == 0) ? (qs[i] & 0x0F) : (qs[i] >> 4);
        tmp[i] = KVALUES_IQ4NL_D[idx];
    }
    v4i_ld v; __builtin_memcpy(&v, tmp, 16); return v;
}

// TQ2_0: super-block = qs[64], d (f16). 32-sub-block g: half=g/4, l=g%4; byte = half*32 + k, value
// = ((qs[byte] >> (l*2)) & 3) - 1 (ternary, q_w in {-1,0,1}). k=half_off..half_off+15.
template <>
__device__ __forceinline__ v4i_ld decode_w_half<WT_TQ2_0>(const uint8_t* sb, int g, int half) {
    const uint8_t* qs = sb;
    const int hf = g >> 2;           // 0,1
    const int l = g & 3;             // 0..3
    int8_t tmp[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int kk = half + i;     // 0..31 within the sub-block
        const int byte = hf * 32 + kk;
        const int q = (qs[byte] >> (l * 2)) & 3;
        tmp[i] = (int8_t)(q - 1);
    }
    v4i_ld v; __builtin_memcpy(&v, tmp, 16); return v;
}

// MAIN per-(32-block, 16-group) weight scale. `sub16` = the 16-group index within the super-block
// (= g*2 + (half==16)). For SCALE_STEP=32 types both halves of a 32-sub-block return the same value
// (so the merged-accumulate path scales once); for Q6_K (SCALE_STEP=16) each 16-group differs.
template <int WTYPE>
__device__ __forceinline__ float decode_w_mainscale(const uint8_t* sb, int sub16);

template <> __device__ __forceinline__ float decode_w_mainscale<WT_Q8_0>(const uint8_t* sb, int) {
    return __half2float(*reinterpret_cast<const __half*>(sb));
}
template <> __device__ __forceinline__ float decode_w_mainscale<WT_Q4_0>(const uint8_t* sb, int) {
    return __half2float(*reinterpret_cast<const __half*>(sb));
}
// Q4_K: d * sc_g (sc from get_scale_min_k4 for sub-block g = sub16/2).
template <> __device__ __forceinline__ float decode_w_mainscale<WT_Q4_K>(const uint8_t* sb, int sub16) {
    const float d = __half2float(*reinterpret_cast<const __half*>(sb));
    int sc, m; q4k_scale_min(sb + 4, sub16 >> 1, &sc, &m);
    return d * (float)sc;
}
// Q6_K: d * scales[is] where is = the 16-group index sub16 (scales are signed i8, 16 of them).
template <> __device__ __forceinline__ float decode_w_mainscale<WT_Q6_K>(const uint8_t* sb, int sub16) {
    const float d = __half2float(*reinterpret_cast<const __half*>(sb + 208));
    const int8_t* sc = reinterpret_cast<const int8_t*>(sb + 192);
    // sub16 in [0,16): the to_float scale index for sub-block g (=sub16/2), quadrant pair. Q6_K's
    // 16 scales index by (is + 2*quad) inside each 128-half; the 16-group order here is the same
    // sequential 16-element walk to_float uses, so scales[sub16] is the matching per-16 scale.
    return d * (float)sc[sub16];
}
// IQ4_XS: d * (ls_g - 32), ls_g the 6-bit sub-block scale for sub-block g = sub16/2.
template <> __device__ __forceinline__ float decode_w_mainscale<WT_IQ4_XS>(const uint8_t* sb, int sub16) {
    const float d_all = __half2float(*reinterpret_cast<const __half*>(sb));
    const uint16_t scales_h = *reinterpret_cast<const uint16_t*>(sb + 2);
    const uint8_t* scales_l = sb + 4;
    const int ib = sub16 >> 1;
    const int ls = ((scales_l[ib >> 1] >> (4 * (ib & 1))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
    return d_all * (float)(ls - 32);
}
// TQ2_0: single d for the whole super-block (NSC=1, symmetric ternary).
template <> __device__ __forceinline__ float decode_w_mainscale<WT_TQ2_0>(const uint8_t* sb, int) {
    return __half2float(*reinterpret_cast<const __half*>(sb + 64));
}

// SECONDARY per-32-sub-block weight scalar. ASYMMETRIC: dmin * m_g (the min bias scale). SYMMETRIC
// SCALE_STEP=16 (Q6_K): the 2nd-16-group main scale of sub-block g (so the high half scales by it).
// Other symmetric types never read this. `g` = the 32-sub-block index (0..SUBS-1).
template <int WTYPE>
__device__ __forceinline__ float decode_w_secscale(const uint8_t* sb, int g);

template <> __device__ __forceinline__ float decode_w_secscale<WT_Q8_0>(const uint8_t*, int) { return 0.0f; }
template <> __device__ __forceinline__ float decode_w_secscale<WT_Q4_0>(const uint8_t*, int) { return 0.0f; }
// Q4_K: dmin * m_g.
template <> __device__ __forceinline__ float decode_w_secscale<WT_Q4_K>(const uint8_t* sb, int g) {
    const float dmin = __half2float(*reinterpret_cast<const __half*>(sb + 2));
    int sc, m; q4k_scale_min(sb + 4, g, &sc, &m);
    return dmin * (float)m;
}
// Q6_K: the 2nd-16-group scale of sub-block g = the main scale at sub16 = 2*g+1.
template <> __device__ __forceinline__ float decode_w_secscale<WT_Q6_K>(const uint8_t* sb, int g) {
    return decode_w_mainscale<WT_Q6_K>(sb, 2 * g + 1);
}
template <> __device__ __forceinline__ float decode_w_secscale<WT_IQ4_XS>(const uint8_t*, int) { return 0.0f; }
template <> __device__ __forceinline__ float decode_w_secscale<WT_TQ2_0>(const uint8_t*, int) { return 0.0f; }

// Shared int8 tile row stride: 32 data bytes + 4 pad bytes (llama's MMQ_MMA_TILE_X_K_Q8_0 padding,
// which keeps row*stride % 8 == 4). With 16B (b128) fragment loads on 32B rows, consecutive rows
// otherwise land on the same 32 LDS banks; +4B shifts each row by one bank-group so the 16-lane
// fragment loads spread across banks. 20480 B LDS total (2*128*36 + 2*128*4) -> still 3 blocks/CU.
#define SROW 36
// Stage the X (activation) int8 tile + the weight int8 tile (per-type decode) + the per-block
// scales for 32-block BLK into shared buffer BUF. Identical X-half staging to the proven path; the
// WEIGHT tile uses decode_w_half<WTYPE>(super-block, g, half) with g = the sub-block index. Scales
// staged once/block: sWd (main, indexed by 16-group via SROW-free 2-slot rows), sWm (secondary).
#define STGI(BUF, BLK)                                                                          \
    do {                                                                                        \
        const int sb_ = (BLK) / SUBS;            /* on-disk super-block index */                \
        const int g_  = (BLK) % SUBS;            /* sub-block within the super-block */          \
        {                                                                                        \
            const int c = t;                                                                     \
            const int rc = c & 255;                                                              \
            const int r = rc >> 1;                                                               \
            const int half = (rc & 1) * 16;                                                      \
            if (c < 256) {                                                                       \
                const int gn = col_tile + r;                                                     \
                v4i_ld v = {0,0,0,0};                                                            \
                if (gn < N) {                                                                    \
                    v = decode_w_half<WTYPE>(wq + ((size_t)gn * nsblk + sb_) * WBYTES, g_, half);\
                }                                                                                \
                __builtin_memcpy(&sWi[(BUF)][r * SROW + half], &v, 16);                          \
            } else {                                                                             \
                const int gm = row_tile + r;                                                     \
                v4i_ld v = {0,0,0,0};                                                            \
                if (gm < M) {                                                                    \
                    v = *reinterpret_cast<const v4i_ld*>(                                        \
                            &xq[(size_t)gm * K + (BLK) * 32 + half]);                            \
                }                                                                                \
                __builtin_memcpy(&sXi[(BUF)][r * SROW + half], &v, 16);                          \
            }                                                                                    \
        }                                                                                        \
        /* Scales staged to shared ONCE per block (like llama's x_df/y_df), single slot per col so   \
           the LDS footprint matches the proven Q8_0 path exactly. sWd = MAIN scale (the only scale  \
           for STEP=32; the 1st-16 scale for Q6_K). sWm = SECONDARY scalar, allocated only when      \
           needed (dmin*m for ASYM; the 2nd-16 scale for Q6_K) -- otherwise a 1-slot dummy. sxs = the\
           q8_1 per-block activation sum (ASYM only). */                                            \
        if (t < 128) {                                                                           \
            const int gn = col_tile + t;                                                        \
            if (gn < N) {                                                                        \
                const uint8_t* wb = wq + ((size_t)gn * nsblk + sb_) * WBYTES;                    \
                sWd[(BUF)][t] = decode_w_mainscale<WTYPE>(wb, g_ * 2 + 0);                        \
                if constexpr (HAS_SEC) sWm[(BUF)][t] = decode_w_secscale<WTYPE>(wb, g_);          \
            } else {                                                                             \
                sWd[(BUF)][t] = 0.0f; if constexpr (HAS_SEC) sWm[(BUF)][t] = 0.0f;                \
            }                                                                                    \
            const int gm = row_tile + t;                                                        \
            sxd[(BUF)][t] = (gm < M) ? __half2float(xd[(size_t)gm * nsblk32 + (BLK)]) : 0.0f;    \
            if constexpr (!SYMM) {                                                                \
                sxs[(BUF)][t] = (gm < M) ? (float)xs[(size_t)gm * nsblk32 + (BLK)] : 0.0f;        \
            }                                                                                    \
        }                                                                                        \
    } while (0)

// Templated device core: identical for every weight type except the WTYPE-selected decode + the
// compile-time accumulation shape (SYMMETRIC / SCALE_STEP). `xs` is read iff !SYMMETRIC (the q8_1
// per-block activation sum from quantize_q8_1); symmetric types pass any pointer (never derefed).
template <int WTYPE>
__device__ __forceinline__ void qmmq_core(
    const int M,
    const int N,
    const int K,            // multiple of 32
    const int ncol_tiles,   // ceil(N/128)
    const int8_t* __restrict__ xq,   // [M, K] int8 pre-quantized activations
    const __half* __restrict__ xd,   // [M, K/32] f16 per-32-block activation scales
    const uint8_t* __restrict__ wq,  // [N, K] quantized weight super-blocks (WTYPE format)
    __half* __restrict__ y,          // [M, N]
    const int* __restrict__ xs       // [M, K/32] i32 per-32-block activation sum (ASYM only)
) {
    constexpr int WBYTES     = wt_traits<WTYPE>::BYTES;       // on-disk super-block stride
    constexpr int SBLK_ELEMS = wt_traits<WTYPE>::SBLK_ELEMS;  // elems per super-block (32 or 256)
    constexpr int SUBS       = SBLK_ELEMS / 32;               // 32-blocks per super-block
    constexpr bool SYMM      = wt_traits<WTYPE>::SYMMETRIC;
    constexpr int NSC16      = 32 / wt_traits<WTYPE>::SCALE_STEP; // distinct scales per 32-block (1 or 2)
    // A secondary per-col weight scalar is staged iff ASYMMETRIC (dmin*m bias) or 2-scales/32-block
    // (Q6_K). Otherwise the dummy 1-slot arrays keep the LDS footprint identical to the proven path.
    constexpr bool HAS_SEC   = (!SYMM) || (NSC16 == 2);
    constexpr int SECN       = HAS_SEC ? 128 : 1;  // sWm slots/col
    constexpr int XSN        = SYMM ? 1 : 128;     // sxs slots/col (ASYM only)
    const int tile = blockIdx.x;
    const int row_tile = (tile / ncol_tiles) * 128;
    const int col_tile = (tile % ncol_tiles) * 128;
    const int t = threadIdx.x;       // 0..511
    const int lane = t & 31;
    const int warp = t >> 5;         // 0..15
    const int wave_m = warp & 3;     // 0..3 (M sub-tile, x32)
    const int wave_n = warp >> 2;    // 0..3 (N sub-tile, x32)
    const int nsblk32 = K >> 5;          // # of 32-blocks along K (for xq/xd/xs indexing)
    const int nsblk   = K / SBLK_ELEMS;  // # of on-disk super-blocks along K (for wq indexing)
    const int nblk    = nsblk32;         // K-loop is over 32-blocks (WMMA granularity)

    __shared__ __attribute__((aligned(16))) int8_t sXi[2][128 * SROW];
    __shared__ __attribute__((aligned(16))) int8_t sWi[2][128 * SROW];
    __shared__ float sxd[2][128];     // per-row activation scale (proven Q8_0 layout)
    __shared__ float sWd[2][128];     // per-col MAIN weight scale (proven Q8_0 layout)
    __shared__ float sWm[2][SECN];    // per-col secondary scalar (dmin*m ASYM | 2nd-16 scale Q6_K)
    __shared__ float sxs[2][XSN];     // per-row activation block-sum (ASYM only)

    v8f acc00 = {0,0,0,0,0,0,0,0}, acc01 = {0,0,0,0,0,0,0,0};
    v8f acc10 = {0,0,0,0,0,0,0,0}, acc11 = {0,0,0,0,0,0,0,0};

    STGI(0, 0);
    __syncthreads();

    for (int blk = 0; blk < nblk; ++blk) {
        const int cur = blk & 1;
        if (blk + 1 < nblk) {
            STGI(cur ^ 1, blk + 1);
        }
        const int nc0 = wave_n * 32 + (lane & 15);
        const int nc1 = wave_n * 32 + 16 + (lane & 15);
        const int arow0 = (wave_m * 32 + (lane & 15)) * SROW;
        const int arow1 = (wave_m * 32 + 16 + (lane & 15)) * SROW;
        const int brow0 = nc0 * SROW;
        const int brow1 = nc1 * SROW;
        v8i af0, af1, bf0, bf1;
        __builtin_memcpy(&af0, &sXi[cur][arow0], 32);
        __builtin_memcpy(&af1, &sXi[cur][arow1], 32);
        __builtin_memcpy(&bf0, &sWi[cur][brow0], 32);
        __builtin_memcpy(&bf1, &sWi[cur][brow1], 32);
        const v4i* a0h = reinterpret_cast<const v4i*>(&af0);
        const v4i* a1h = reinterpret_cast<const v4i*>(&af1);
        const v4i* b0h = reinterpret_cast<const v4i*>(&bf0);
        const v4i* b1h = reinterpret_cast<const v4i*>(&bf1);
        const int xrow = wave_m * 32;
        if constexpr (NSC16 == 1) {
            // ONE scale per 32-block (Q8_0/Q4_0/Q4_K/IQ4_XS/TQ2_0): both WMMA halves accumulate
            // into the SAME int32, scaled once. For SYMMETRIC this is byte-identical to the proven
            // Q8_0 path; ASYMMETRIC adds the -dmin*m*d_x*ix bias.
            v8i ic00 = {0,0,0,0,0,0,0,0}, ic01 = {0,0,0,0,0,0,0,0};
            v8i ic10 = {0,0,0,0,0,0,0,0}, ic11 = {0,0,0,0,0,0,0,0};
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                ic00 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0h[h], true, b0h[h], ic00, false);
                ic01 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0h[h], true, b1h[h], ic01, false);
                ic10 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1h[h], true, b0h[h], ic10, false);
                ic11 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1h[h], true, b1h[h], ic11, false);
            }
            const float wd0 = sWd[cur][nc0];
            const float wd1 = sWd[cur][nc1];
            const float wm0 = SYMM ? 0.0f : sWm[cur][nc0 % SECN];
            const float wm1 = SYMM ? 0.0f : sWm[cur][nc1 % SECN];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float xd0 = sxd[cur][xrow + i * 2 + (lane >> 4)];
                const float xd1 = sxd[cur][xrow + 16 + i * 2 + (lane >> 4)];
                acc00[i] += (float)ic00[i] * xd0 * wd0;
                acc01[i] += (float)ic01[i] * xd0 * wd1;
                acc10[i] += (float)ic10[i] * xd1 * wd0;
                acc11[i] += (float)ic11[i] * xd1 * wd1;
                if constexpr (!SYMM) {
                    const float xs0 = sxs[cur][xrow + i * 2 + (lane >> 4)];
                    const float xs1 = sxs[cur][xrow + 16 + i * 2 + (lane >> 4)];
                    acc00[i] -= xd0 * xs0 * wm0;
                    acc01[i] -= xd0 * xs0 * wm1;
                    acc10[i] -= xd1 * xs1 * wm0;
                    acc11[i] -= xd1 * xs1 * wm1;
                }
            }
        } else {
            // TWO scales per 32-block (Q6_K, SCALE_STEP=16, SYMMETRIC): each 16-half scales by its
            // own weight scale, so the halves accumulate into SEPARATE int32s and are scaled apart.
            const float wd0a = sWd[cur][nc0], wd0b = sWm[cur][nc0 % SECN];
            const float wd1a = sWd[cur][nc1], wd1b = sWm[cur][nc1 % SECN];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                v8i ic00 = {0,0,0,0,0,0,0,0}, ic01 = {0,0,0,0,0,0,0,0};
                v8i ic10 = {0,0,0,0,0,0,0,0}, ic11 = {0,0,0,0,0,0,0,0};
                ic00 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0h[h], true, b0h[h], ic00, false);
                ic01 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0h[h], true, b1h[h], ic01, false);
                ic10 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1h[h], true, b0h[h], ic10, false);
                ic11 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1h[h], true, b1h[h], ic11, false);
                const float wd0 = (h == 0) ? wd0a : wd0b;
                const float wd1 = (h == 0) ? wd1a : wd1b;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float xd0 = sxd[cur][xrow + i * 2 + (lane >> 4)];
                    const float xd1 = sxd[cur][xrow + 16 + i * 2 + (lane >> 4)];
                    acc00[i] += (float)ic00[i] * xd0 * wd0;
                    acc01[i] += (float)ic01[i] * xd0 * wd1;
                    acc10[i] += (float)ic10[i] * xd1 * wd0;
                    acc11[i] += (float)ic11[i] * xd1 * wd1;
                }
            }
        }
        __syncthreads();
    }

    const int lcol = lane & 15;
    const int lrow = lane >> 4;
    const int rbase = row_tile + wave_m * 32;
    const int cbase = col_tile + wave_n * 32;
    const int c0 = cbase + lcol;
    const int c1 = cbase + 16 + lcol;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r0 = rbase + i * 2 + lrow;
        const int r1 = rbase + 16 + i * 2 + lrow;
        if (c0 < N) {
            if (r0 < M) y[(size_t)r0 * N + c0] = __float2half(acc00[i]);
            if (r1 < M) y[(size_t)r1 * N + c0] = __float2half(acc10[i]);
        }
        if (c1 < N) {
            if (r0 < M) y[(size_t)r0 * N + c1] = __float2half(acc01[i]);
            if (r1 < M) y[(size_t)r1 * N + c1] = __float2half(acc11[i]);
        }
    }
}

// Per-type entry points. Each instantiates the ONE shared core with its WTYPE; the launcher
// dispatches on the weight GgmlDType. Q8_0/Q4_0 stay the proven SYMMETRIC NSC16==1 SUBS==1 path
// (the `if constexpr` branches elide -> byte-identical codegen, `xs` unused). Q4_K is ASYMMETRIC
// (min bias via xs); Q6_K/IQ4_XS/TQ2_0 are symmetric super-block types. Adding one = one WTYPE row.
#define DEFINE_QMMQ(NAME, WTYPE)                                                                 \
    extern "C" __global__ void qmmq_##NAME##_f16(                                                \
        const int M, const int N, const int K, const int ncol_tiles,                             \
        const int8_t* __restrict__ xq, const __half* __restrict__ xd,                            \
        const uint8_t* __restrict__ wq, __half* __restrict__ y,                                  \
        const int* __restrict__ xs) {                                                            \
        qmmq_core<WTYPE>(M, N, K, ncol_tiles, xq, xd, wq, y, xs);                                 \
    }

DEFINE_QMMQ(q8_0,  WT_Q8_0)
DEFINE_QMMQ(q4_0,  WT_Q4_0)
DEFINE_QMMQ(q4k,   WT_Q4_K)
DEFINE_QMMQ(q6k,   WT_Q6_K)
DEFINE_QMMQ(iq4xs, WT_IQ4_XS)
DEFINE_QMMQ(tq2_0, WT_TQ2_0)

#undef DEFINE_QMMQ
#undef STGI
#undef SROW
#undef STG
#undef BK
#undef WT_Q8_0
#undef WT_Q4_0
#undef WT_Q4_K
#undef WT_Q6_K
#undef WT_IQ4_XS
#undef WT_TQ2_0
#endif
