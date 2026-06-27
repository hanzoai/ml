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

// f32 activation variant of quantize_q8 (symmetric): reads float instead of __half. MoE prefill
// activations arrive in f32 (the model's compute dtype); quantizing f32 directly skips the f32->f16
// to_dtype cast (a 16.7M-elem read+write per gate/up) the GEMM glue used to pay before this kernel.
extern "C" __global__ void quantize_q8_f32(
    const int M,
    const int K,
    const float* __restrict__ x,
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
    const float v = x[idx];
    float a = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a = fmaxf(a, __shfl_xor(a, off));
    }
    const float inv = (a > 0.0f) ? (127.0f / a) : 0.0f;
    int q = (int)roundf(v * inv);
    q = max(-127, min(127, q));
    xq[idx] = (int8_t)q;
    if (lane == 0) {
        xd[(size_t)m * nblk + blk] = __float2half(a / 127.0f);
    }
}

// f32 activation variant of quantize_q8_1 (asymmetric: also emits the per-block int8 sum xs).
extern "C" __global__ void quantize_q8_1_f32(
    const int M,
    const int K,
    const float* __restrict__ x,
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
    const float v = x[idx];
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

// Fused MoE expert-combine: out[i, j] = sum_{e<topk} scores[i, e] * ys[i, e, j].
// Replaces the engine's broadcast_mul + sum(Minus2): that path cast ys -> f32 (a 16.7M-elem
// cast), wrote a 16.7M-elem f32 product temp, then ran an 8-wide reduce at 2M blocks x 8 threads
// (24/32 lanes idle, dst-strided). This reads ys ONCE (coalesced over j), accumulates topk in f32,
// and writes out[i, j] -- one launch, one pass, no temp. NOT a qmmq_core variant (no matrix cores,
// no weight decode) -- a standalone glue-reduction kernel.
template <typename T>
__device__ __forceinline__ void moe_combine_core(
    const int ntok, const int topk, const int N,
    const T* __restrict__ ys,         // [ntok, topk, N]
    const float* __restrict__ scores, // [ntok, topk]
    T* __restrict__ out               // [ntok, N]
) {
    const size_t total = (size_t)ntok * N;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x; gid < total; gid += stride) {
        const int i = (int)(gid / N);
        const int j = (int)(gid - (size_t)i * N);
        const float* __restrict__ sc = scores + (size_t)i * topk;
        const T* __restrict__ yp = ys + ((size_t)i * topk) * N + j;
        float acc = 0.0f;
        for (int e = 0; e < topk; ++e) {
            acc += sc[e] * (float)yp[(size_t)e * N];
        }
        out[gid] = (T)acc;
    }
}

extern "C" __global__ void moe_combine_f16(
    const int ntok, const int topk, const int N,
    const __half* __restrict__ ys, const float* __restrict__ scores, __half* __restrict__ out
) { moe_combine_core<__half>(ntok, topk, N, ys, scores, out); }

extern "C" __global__ void moe_combine_bf16(
    const int ntok, const int topk, const int N,
    const hip_bfloat16* __restrict__ ys, const float* __restrict__ scores, hip_bfloat16* __restrict__ out
) { moe_combine_core<hip_bfloat16>(ntok, topk, N, ys, scores, out); }

extern "C" __global__ void moe_combine_f32(
    const int ntok, const int topk, const int N,
    const float* __restrict__ ys, const float* __restrict__ scores, float* __restrict__ out
) { moe_combine_core<float>(ntok, topk, N, ys, scores, out); }

// Fused MoE router: ONE kernel replacing the softmax->sort->narrow->sum->div chain that the
// per-token router previously issued as ~6 launches/layer. One block per token reduces the F32
// router logits [ntok, n_experts] to the selected expert ids [ntok, topk] (descending logit) and
// their softmax weights [ntok, topk]; `norm` renormalizes the topk weights to sum 1 (norm_topk_prob).
// Weight = exp(l - max)/Z over ALL experts (full softmax denominator, so norm=0 matches a plain
// softmax-then-topk), so this is numerically faithful to the chain it replaces.
#define MOE_ROUTE_MAX_E 256
#define MOE_ROUTE_MAX_K 32
#define MOE_ROUTE_THREADS 64
#define MOE_ROUTE_NEG_INF (-3.402823466e38f)

extern "C" __global__ __launch_bounds__(MOE_ROUTE_THREADS) void moe_route(
    const int ntok, const int n_experts, const int topk, const int norm,
    const float* __restrict__ logits,   // [ntok, n_experts]
    unsigned int* __restrict__ out_ids, // [ntok, topk]
    float* __restrict__ out_w           // [ntok, topk]
) {
    const int tok = blockIdx.x;
    if (tok >= ntok) return;
    const int t = threadIdx.x;
    __shared__ float sl[MOE_ROUTE_MAX_E];
    __shared__ float se[MOE_ROUTE_MAX_E];
    __shared__ float red[MOE_ROUTE_THREADS];
    __shared__ int rid[MOE_ROUTE_THREADS];
    __shared__ float sp[MOE_ROUTE_MAX_K];
    __shared__ int sidx[MOE_ROUTE_MAX_K];
    const float* __restrict__ lp = logits + (size_t)tok * n_experts;
    for (int e = t; e < n_experts; e += MOE_ROUTE_THREADS) sl[e] = lp[e];
    __syncthreads();

    float m = MOE_ROUTE_NEG_INF;
    for (int e = t; e < n_experts; e += MOE_ROUTE_THREADS) m = fmaxf(m, sl[e]);
    red[t] = m;
    __syncthreads();
    for (int s = MOE_ROUTE_THREADS / 2; s > 0; s >>= 1) {
        if (t < s) red[t] = fmaxf(red[t], red[t + s]);
        __syncthreads();
    }
    const float mx = red[0];
    __syncthreads();

    float zp = 0.0f;
    for (int e = t; e < n_experts; e += MOE_ROUTE_THREADS) {
        const float v = expf(sl[e] - mx);
        se[e] = v;
        zp += v;
    }
    red[t] = zp;
    __syncthreads();
    for (int s = MOE_ROUTE_THREADS / 2; s > 0; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        __syncthreads();
    }
    const float Z = red[0];
    __syncthreads();

    // Parallel top-k: each round is a block-wide argmax (max logit, lowest index on tie) over the
    // not-yet-selected experts. Selection is tracked by clobbering the chosen logit to -inf in `sl`.
    for (int r = 0; r < topk; ++r) {
        float lv = MOE_ROUTE_NEG_INF;
        int li = n_experts;
        for (int e = t; e < n_experts; e += MOE_ROUTE_THREADS) {
            const float v = sl[e];
            if (v > lv || (v == lv && e < li)) { lv = v; li = e; }
        }
        red[t] = lv;
        rid[t] = li;
        __syncthreads();
        for (int s = MOE_ROUTE_THREADS / 2; s > 0; s >>= 1) {
            if (t < s) {
                const float ov = red[t + s];
                const int oi = rid[t + s];
                if (ov > red[t] || (ov == red[t] && oi < rid[t])) { red[t] = ov; rid[t] = oi; }
            }
            __syncthreads();
        }
        const int best = rid[0];
        if (t == 0) {
            sidx[r] = best;
            sp[r] = se[best] / Z;
        }
        __syncthreads();
        if (t == 0) sl[best] = MOE_ROUTE_NEG_INF;
        __syncthreads();
    }

    if (t == 0) {
        float inv = 1.0f;
        if (norm) {
            float wsum = 0.0f;
            for (int r = 0; r < topk; ++r) wsum += sp[r];
            inv = 1.0f / wsum;
        }
        for (int r = 0; r < topk; ++r) {
            out_ids[(size_t)tok * topk + r] = (unsigned int)sidx[r];
            out_w[(size_t)tok * topk + r] = norm ? sp[r] * inv : sp[r];
        }
    }
}
#undef MOE_ROUTE_MAX_E
#undef MOE_ROUTE_MAX_K
#undef MOE_ROUTE_THREADS
#undef MOE_ROUTE_NEG_INF

// get_scale_min_k4 scale/min unpack -- DEFINED below (reused by the scalar Q4_K path + the unified
// core); forward-declared here so the dp4a Q4_K kernel can call the SAME unpack (one source of
// truth for the 6-bit scale layout).
__device__ __forceinline__ void q4k_scale_min(const uint8_t* __restrict__ s, int j, int* sc, int* m);

// get_scale_min_k4 over the 12 scale bytes held in 3 PACKED WORDS (not DRAM) -- forward-declared so the
// dp4a Q4_K kernel reads the whole 16-byte header (d/dmin + 12 scales) in ONE 128-bit load and unpacks
// the scales from registers instead of byte-granular gathers. DEFINED next to q4k_scale_min (one source
// of truth for the 6-bit scale layout); BIT-IDENTICAL to it, only the byte source changes.
__device__ __forceinline__ void q4k_scale_min_words(uint32_t w0, uint32_t w1, uint32_t w2, int j, int* sc, int* m);

// q3k_scales unpack -- DEFINED below (reused by the scalar Q3_K path + the dp4a Q3_K kernel);
// forward-declared here so both call the SAME 16-scale reconstruction (one source of truth).
__device__ __forceinline__ void q3k_scales(const uint8_t* __restrict__ s, int8_t* __restrict__ out);

// Decode-side weight type ids -- the SINGLE shared id space for BOTH the int8-dp4a core
// (`qdp4a<WTYPE>` / `qmatvec_dp4a_core`) and the scalar core (`qdec<WTYPE>` / `qmatvec_core`); the
// Rust `RocmQuantType` selects the dp4a vs scalar entry point per type. Distinct from the prefill
// WT_* id space (#undef'd at end-of-file). #undef'd at end-of-file. BYTES = on-disk block stride,
// ELEMS = weights per block, SYMMETRIC = which accumulation shape (the partial-dot/block contract
// folds min/delta inside the per-type decode, so the cores stay shape-agnostic).
#define DW_Q8_0   0   // 34 B,  32 elems, symmetric 8-bit.            val = d*qs[p]
#define DW_Q4_0   1   // 18 B,  32 elems, symmetric nibble (-8).      val = d*((nib)-8)
#define DW_Q4_K   2   // 144 B, 256 elems, ASYMMETRIC super-block.    val = d*sc*q - dmin*m
#define DW_Q6_K   3   // 210 B, 256 elems, symmetric K-quant (6-bit). val = d*sc*(q-32)
#define DW_IQ4_XS 4   // 136 B, 256 elems, symmetric codebook (LUT).  val = d*(ls-32)*KVALUES[idx]
#define DW_TQ2_0  5   // 66 B,  256 elems, symmetric ternary (2-bit). val = d*((q&3)-1)
#define DW_Q2_K   6   // 84 B,  256 elems, ASYMMETRIC k-quant (2-bit). val = d*sc*q - dmin*m (q 0..3)
#define DW_Q3_K   7   // 110 B, 256 elems, symmetric k-quant (3-bit). val = d*sc*(q-4) (q 0..7, sc signed)
#define DW_Q5_K   8   // 176 B, 256 elems, ASYMMETRIC super-block.    val = d*sc*(nib|hi<<4) - dmin*m
#define DW_Q4_1   9   // 20 B,  32 elems,  ASYMMETRIC legacy.         val = d*nib + m
#define DW_Q5_0   10   // 22 B,  32 elems,  symmetric legacy 5-bit.    val = d*((nib|hi<<4)-16)
#define DW_Q5_1   11   // 24 B,  32 elems,  ASYMMETRIC legacy 5-bit.   val = d*(nib|hi<<4) + m
#define DW_Q8_1   12  // 36 B,  32 elems,  symmetric 8-bit (+sum).    val = d*qs[p]
#define DW_IQ2_XXS 13  // 66 B,  256 elems, SYM 2.06bpw codebook.  val = db*grid_u8*sign (ksigns)
#define DW_IQ2_XS  14  // 74 B,  256 elems, SYM 2.31bpw codebook.  + per-32 scales nibble (ksigns)
#define DW_IQ2_S   15  // 82 B,  256 elems, SYM 2.56bpw codebook.  signs in qs upper half
#define DW_IQ3_XXS 16  // 98 B,  256 elems, SYM 3.06bpw codebook.  u32 grid, *0.5 scale (ksigns)
#define DW_IQ3_S   17 // 110 B, 256 elems, SYM 3.31bpw codebook.  u32 grid, d*(1+2*sc), signs field
#define DW_IQ4_NL 18   // 18 B,  32 elems,  symmetric codebook (LUT).  val = d*KVALUES[idx]
#define DW_TQ1_0  19   // 54 B,  256 elems, symmetric ternary base-3 (1.69 bpw). val = d*(xi-1)
#define DW_IQ1_S  20   // 50 B,  256 elems, signed grid + delta bias.  val = dl*(grid_i8 + delta)
#define DW_IQ1_M  21   // 56 B,  256 elems, signed grid + delta bias, scale from 4 nibbles.
#define DW_MXFP4  22   // 17 B,  32 elems,  E8M0 scale + FP4 codebook (gpt-oss). val = e8m0_half(e)*KVALUES_MXFP4[idx]

// Type-dispatched f32 -> output store (the only place the element type leaks): __half needs
// __float2half, hip_bfloat16 takes float in its ctor. Keeps both cores' epilogue type-agnostic.
__device__ __forceinline__ void qstore(__half* p, float v)        { *p = __float2half(v); }
__device__ __forceinline__ void qstore(hip_bfloat16* p, float v)  { *p = hip_bfloat16(v); }
__device__ __forceinline__ void qstore(float* p, float v)         { *p = v; }

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
// UNIFIED int8 dp4a DECODE matvec core. ONE quant-agnostic warp-per-row dp4a core
// (`qmatvec_dp4a_core<WTYPE,XT>`) + ONE per-type packed-int decode (`qdp4a<WTYPE>::block`) covers the
// symmetric/asymmetric K-quants whose 256-element super-block has a faithful dp4a path (Q4_K now,
// Q6_K now, Q5_K-ready). It is the int8-SIMD twin of the scalar `qmatvec_core<WTYPE>`: instead of
// dequantizing each weight to f32 and doing float MACs, the activation row is pre-quantized ONCE to
// q8_1 (int8 qs + per-32-block f16 scale d8) and each weight is dotted against the q8_1 int8 via
// `v_dot4` (4x int8 MAC/instruction), then scaled -- exactly llama.cpp's vec_dot_*_q8_1 technique.
// The Rust side routes dp4a-capable types here (RocmQuantType::dp4a_capable) and every other type to
// the scalar core; there is ONE dp4a core and ONE scalar fallback, selected by the trait -- no
// per-quant dp4a kernel.
//
// WARP-per-row, LANE-STRIDED over super-blocks (matches the scalar decode core): lane L streams whole
// super-blocks b = L, L+32, ... so 32 independent super-block loads are in flight (hides LPDDR5X
// latency on the m=1 FFN shapes), each lane fully int8-decodes its super-blocks, then a warp-shuffle
// reduces the per-lane partials. The q8_1 activation (xq int8 + xd f16 scale) is laid out [k]
// contiguous: super-block b, 32-block j -> q8 ints at xq + (b*8 + j)*32, scale xd[b*8 + j].
//
// Per-type `qdp4a<WTYPE>::block(blk, xq8, xd8)` returns the COMPLETE f32 dot of ONE super-block (256
// weights) against its q8_1 activation (256 int8 `xq8` = 8 blocks of 32 contiguous, 8 f16 scales
// `xd8`). It mirrors the SAME bit layout as `qdec<WTYPE>::partial` (the scalar CPU oracle), so the
// dp4a output is bit-faithful (only the f32 accumulation order changes, which the 1%-magnitude
// numeric gate explicitly allows). Summing all 8 sub-blocks in one loop is algebraically identical to
// llama's grid-tiled iqs/bq8_offset machinery (same per-element products, same f32 accumulation).

template <int WTYPE> struct qdp4a;

// Q4_K (ASYMMETRIC). block_q4_K = {d, dmin, 12 scale bytes, 128 qs}. For sub-block j in [0,8):
// (sc[j], m[j]) = get_scale_min_k4(scales, j); chunk c = j/2 covers qs[c*32, c*32+32), sub-block 2c =
// LOW nibble, 2c+1 = HIGH nibble (llama's v0i=(v>>0)&0x0F0F0F0F / v1i=(v>>4)&0x0F0F0F0F split = the
// to_float chunk layout). q8_1 block j has int8 u[j][0..31] (8 ints) + scale d8[j]; per impl_vmmq:
//   dot1 = sum_k nibble_k * u_k (8 dp4a); dot2 = sum_k u_k (8 dp4a of 0x01010101 = q8_1 sum)
//   sumf_d += d8[j]*(dot1*sc[j]);  sumf_m += d8[j]*(dot2*m[j]);  result = d*sumf_d - dmin*sumf_m
template <> struct qdp4a<DW_Q4_K> {
    static constexpr int NSUB = 4;           // 4 chunks per super-block, each independently summable
    // Chunk `sub` (0..3) of one super-block: qs[sub*32, sub*32+32) = sub-blocks 2*sub (low) + 2*sub+1
    // (high). Returns the chunk's COMPLETE f32 contribution (already folded into d/dmin units) so
    // chunk partials sum directly to `block()`. This is the per-sub-unit work item the MoE core
    // strides across lanes -- 4 chunks * nblocks fills the warp on the narrow-k expert shapes.
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int sub) {
        // ONE 128-bit header load: blk[0,16) = d(2) + dmin(2) + 12 scale bytes -> registers. blk is
        // 16-byte aligned (144 = 16*9, weight base >=256 B aligned), so this is a single coalesced
        // global_load_b128 instead of two f16 loads + 10 byte-granular scale gathers -- the load-side
        // gap vs Q6_K. hdr.x = {d, dmin}; hdr.y/z/w = scale bytes 0-3/4-7/8-11. The scale unpack then
        // runs on the packed words (q4k_scale_min_words), BIT-IDENTICAL to the byte-pointer unpack.
        const uint4 hdr = *reinterpret_cast<const uint4*>(blk);
        const float d    = __half2float(__ushort_as_half((unsigned short)(hdr.x & 0xFFFFu)));
        const float dmin = __half2float(__ushort_as_half((unsigned short)(hdr.x >> 16)));
        const int* q4 = reinterpret_cast<const int*>(blk + 16);
        const int* u  = reinterpret_cast<const int*>(xq8);
        int sc_lo, m_lo, sc_hi, m_hi;
        q4k_scale_min_words(hdr.y, hdr.z, hdr.w, 2 * sub,     &sc_lo, &m_lo);
        q4k_scale_min_words(hdr.y, hdr.z, hdr.w, 2 * sub + 1, &sc_hi, &m_hi);
        const int* q4c   = q4 + sub * 8;
        const int* u_lo  = u + (2 * sub)     * 8;
        const int* u_hi  = u + (2 * sub + 1) * 8;
        const float d8_lo = __half2float(xd8[2 * sub]);
        const float d8_hi = __half2float(xd8[2 * sub + 1]);
        int dot1_lo = 0, dot2_lo = 0, dot1_hi = 0, dot2_hi = 0;
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int w   = q4c[t];
            const int wlo = (w >> 0) & 0x0F0F0F0F;
            const int whi = (w >> 4) & 0x0F0F0F0F;
            dot1_lo = hip_dp4a(wlo, u_lo[t], dot1_lo);
            dot2_lo = hip_dp4a(0x01010101, u_lo[t], dot2_lo);
            dot1_hi = hip_dp4a(whi, u_hi[t], dot1_hi);
            dot2_hi = hip_dp4a(0x01010101, u_hi[t], dot2_hi);
        }
        const float sumf_d = d8_lo * (dot1_lo * sc_lo) + d8_hi * (dot1_hi * sc_hi);
        const float sumf_m = d8_lo * (dot2_lo * m_lo)  + d8_hi * (dot2_hi * m_hi);
        return d * sumf_d - dmin * sumf_m;
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int sub = 0; sub < NSUB; ++sub) s += partial(blk, xq8, xd8, sub);
        return s;
    }
};

// Q5_K (ASYMMETRIC, 5-bit). block_q5_K = {d, dmin, 12 scale bytes, qh[32], qs[128]} = 176 B. The
// 5th bit of each weight rides qh: for sub-block e (0..8), position l (0..32), the high bit is
// qh[l] bit e (low sub-block 2*c uses bit 2*c, high sub-block 2*c+1 uses bit 2*c+1 -- exactly the
// to_float u1/u2<<2 walk). So Q5_K = Q4_K with q in [0,31] instead of [0,15]: chunk c (=sub) covers
// qs[c*32, c*32+32), low nibble + qh bit 2c -> sub-block 2c, high nibble + qh bit 2c+1 -> sub-block
// 2c+1. The 5-bit code (nib | bit<<4) in [0,31] is a valid signed int8, so dp4a(q5, u) is exact;
// the asymmetric min term (dot2 = sum_k u, scaled by m) is identical to Q4_K's.
template <> struct qdp4a<DW_Q5_K> {
    static constexpr int NSUB = 4;
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int sub) {
        const float d    = __half2float(*reinterpret_cast<const __half*>(blk));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(blk + 2));
        const uint8_t* scales = blk + 4;
        const int* qh = reinterpret_cast<const int*>(blk + 16);  // 8 ints = 32 high-bit bytes
        const int* q4 = reinterpret_cast<const int*>(blk + 48);  // 128 quant bytes
        const int* u  = reinterpret_cast<const int*>(xq8);
        int sc_lo, m_lo, sc_hi, m_hi;
        q4k_scale_min(scales, 2 * sub,     &sc_lo, &m_lo);
        q4k_scale_min(scales, 2 * sub + 1, &sc_hi, &m_hi);
        const int* q4c   = q4 + sub * 8;
        const int* u_lo  = u + (2 * sub)     * 8;
        const int* u_hi  = u + (2 * sub + 1) * 8;
        const float d8_lo = __half2float(xd8[2 * sub]);
        const float d8_hi = __half2float(xd8[2 * sub + 1]);
        const int blo = 2 * sub;        // qh bit for the low sub-block
        const int bhi = 2 * sub + 1;    // qh bit for the high sub-block
        int dot1_lo = 0, dot2_lo = 0, dot1_hi = 0, dot2_hi = 0;
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int w  = q4c[t];
            const int qv = qh[t];       // 4 qh bytes for positions l=t*4..t*4+4
            const int wlo = ((w >> 0) & 0x0F0F0F0F) | (((qv >> blo) & 0x01010101) << 4);
            const int whi = ((w >> 4) & 0x0F0F0F0F) | (((qv >> bhi) & 0x01010101) << 4);
            dot1_lo = hip_dp4a(wlo, u_lo[t], dot1_lo);
            dot2_lo = hip_dp4a(0x01010101, u_lo[t], dot2_lo);
            dot1_hi = hip_dp4a(whi, u_hi[t], dot1_hi);
            dot2_hi = hip_dp4a(0x01010101, u_hi[t], dot2_hi);
        }
        const float sumf_d = d8_lo * (dot1_lo * sc_lo) + d8_hi * (dot1_hi * sc_hi);
        const float sumf_m = d8_lo * (dot2_lo * m_lo)  + d8_hi * (dot2_hi * m_hi);
        return d * sumf_d - dmin * sumf_m;
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int sub = 0; sub < NSUB; ++sub) s += partial(blk, xq8, xd8, sub);
        return s;
    }
};

// Q6_K (SYMMETRIC 6-bit). block_q6_K = {ql[128], qh[64], scales[16] signed i8, d (f16)} = 210 B.
// The CPU oracle (k_quants BlockQ6K::vec_dot / qdec<DW_Q6_K>) dequantizes to aux8[256] then dots
// scales[p/16] * aux8[p] * q8[p] for p in [0,256): aux8 position p = half*128 + quad*32 + l
// (half=p/128, quad=(p%128)/32, l=p%32), q8 (q8_1) index = p, weight scale index = p/16. So the q8_1
// activation index for aux8[p] is just p (the contiguous xq8 layout), scale = scales[p/16]. The 6-bit
// weight aux8[p] = (reconstructed-from-ql/qh) - 32 lands in [-32,31] = signed int8 directly, so the
// `v_dot4` runs the centered weight against the int8 q8_1 with NO __vsubss4 (absent on ROCm 7.13);
// we reconstruct 4 consecutive centered weights into one packed int and dp4a against *(int*)(xq8+p).
// 4 consecutive p share one scale iff same p/16, which holds within each 16-group (p..p+15) -> 4 dp4a
// of 4 per 16-group. d8 (q8_1 32-block scale) for p is xd8[p/32]. result = d * sum_{16-grp g}
// scales[g] * d8[g/2] * (sum of the group's 4 dp4a). Bit-faithful to the oracle (f32 reorder only).
template <> struct qdp4a<DW_Q6_K> {
    static constexpr int NSUB = 16;          // 16 scale-groups per super-block, each summable
    // Scale-group `sub` (0..15) of one super-block: positions p = sub*16 .. sub*16+15. Returns the
    // group's COMPLETE f32 contribution (d folded in) so group partials sum directly to `block()`.
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int sub) {
        const uint8_t* ql_b = blk;                                       // 128
        const uint8_t* qh_b = blk + 128;                                 // 64
        const int8_t*  sc_b = reinterpret_cast<const int8_t*>(blk + 192);// 16 signed
        const float d = __half2float(*reinterpret_cast<const __half*>(blk + 208));
        const int g     = sub;
        const int sc    = sc_b[g];
        const float d8  = __half2float(xd8[g >> 1]);   // q8_1 32-block scale: 2 16-groups per block
        // A 16-element scale-group lies wholly inside ONE 32-quad, so half/quad/l0 are constant for
        // all 16 positions (p0>>7, (p0>>5)&3, p0&31 with p0=g*16 const within the group). Load the
        // group's 16 ql + 16 qh bytes as four packed ints each (vectorized, vs 16 byte loads) and
        // reconstruct 4 weights per int with packed nibble/2-bit ops -- no per-element branch, no
        // byte-granular gathers. The quad/half selection is the ONE branch, now per-group not per-elem.
        const int base = g * 16;
        const int half = base >> 7;
        const int quad = (base >> 5) & 3;
        const int l0   = base & 31;                      // 0 or 16
        const int* ql4 = reinterpret_cast<const int*>(ql_b + 64 * half + l0 + ((quad & 1) ? 32 : 0));
        const int* qh4 = reinterpret_cast<const int*>(qh_b + 32 * half + l0);
        const int hsh  = quad * 2;                        // qh shift: 0,2,4,6 for quad 0,1,2,3
        const bool loNib = (quad < 2);                    // quad 0/1 = low nibble, 2/3 = high
        // Handle the q6-32 centering as a bias term (like Q4_K's dot2): the raw 6-bit code q in [0,63]
        // is a valid signed int8, so dp4a(q_packed, u) is exact; sum_p (q_p-32)*u_p = dot_q - 32*sum_u.
        // sum_u via dp4a(0x01010101, u) avoids the cross-byte-borrow that a packed (q-32) subtract has.
        int dot_q = 0, sum_u = 0;
        #pragma unroll
        for (int q = 0; q < 4; ++q) {        // 4 dp4a of 4 contiguous positions p0 = base + q*4
            const int lv = ql4[q];
            const int hv = qh4[q];
            const int nib = loNib ? ((lv >> 0) & 0x0F0F0F0F) : ((lv >> 4) & 0x0F0F0F0F);
            const int hi  = ((hv >> hsh) & 0x03030303) << 4;
            const int wpk = nib | hi;                     // raw q6 in [0,63] per byte = valid int8
            const int u = *reinterpret_cast<const int*>(xq8 + base + q * 4);
            dot_q = hip_dp4a(wpk, u, dot_q);
            sum_u = hip_dp4a(0x01010101, u, sum_u);
        }
        return d * d8 * (float)((dot_q - 32 * sum_u) * sc);
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int sub = 0; sub < NSUB; ++sub) s += partial(blk, xq8, xd8, sub);
        return s;
    }
};

// Q2_K (ASYMMETRIC 2-bit). block_q2_K = {scales[16], qs[64], d (f16), dmin (f16)} = 84 B. The 256
// elems are 8 32-blocks J; 32-block J holds TWO 16-elem sub-blocks: lo (scale 2J) + hi (scale 2J+1),
// each scales[is] packing a 4-bit scale (&0xF) + 4-bit min (>>4). qs byte (J>>2)*32 + r packs the
// 2-bit quant of element r of 32-block J at bit (2*(J&3)). q in [0,3] is a valid int8 so dp4a runs it
// directly; the per-sub-block min rides the dp4a(0x01010101,u) sum (like Q4_K's dot2). NSUB=8: one
// 32-block (lo+hi) per sub-unit = one q8_1 32-scale d8[J], so the warp fills on narrow-k expert shapes.
template <> struct qdp4a<DW_Q2_K> {
    static constexpr int NSUB = 8;
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int sub) {
        const uint8_t* scales = blk;                                        // 16
        const int* qs = reinterpret_cast<const int*>(blk + 16);            // 64 B = 16 ints
        const float d    = __half2float(*reinterpret_cast<const __half*>(blk + 80));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(blk + 82));
        const int J = sub;                                                  // 32-block index 0..7
        const int shift = 2 * (J & 3);
        const int* qint = qs + (J >> 2) * 8;                               // 8 ints for this 32-block
        const int* u    = reinterpret_cast<const int*>(xq8) + J * 8;       // 8 q8_1 ints
        const float d8  = __half2float(xd8[J]);
        const uint8_t sc_lo = scales[2 * J];
        const uint8_t sc_hi = scales[2 * J + 1];
        int dlo = 0, mlo = 0, dhi = 0, mhi = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {                                       // lo 16: ints 0..3 (scale 2J)
            const int q = (qint[i] >> shift) & 0x03030303;
            dlo = hip_dp4a(q, u[i], dlo);
            mlo = hip_dp4a(0x01010101, u[i], mlo);
        }
        #pragma unroll
        for (int i = 4; i < 8; ++i) {                                       // hi 16: ints 4..7 (scale 2J+1)
            const int q = (qint[i] >> shift) & 0x03030303;
            dhi = hip_dp4a(q, u[i], dhi);
            mhi = hip_dp4a(0x01010101, u[i], mhi);
        }
        const float sumf_d = d8 * (float)(dlo * (sc_lo & 0xF) + dhi * (sc_hi & 0xF));
        const float sumf_m = d8 * (float)(mlo * (sc_lo >> 4)  + mhi * (sc_hi >> 4));
        return d * sumf_d - dmin * sumf_m;
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int sub = 0; sub < NSUB; ++sub) s += partial(blk, xq8, xd8, sub);
        return s;
    }
};

// Q3_K (SYMMETRIC 3-bit, signed per-16 scale). block_q3_K = {hmask[32], qs[64], scales[12], d (f16)}
// = 110 B. Element r of 32-block J: low2 = (qs[(J>>2)*32 + r] >> (2*(J&3))) & 3; hbit = (hmask[r]>>J)&1;
// q = (low2 | hbit<<2) - 4 in [-4,3]. raw = low2|hbit<<2 in [0,7] is a valid int8 so dp4a runs it; the
// -4 centering rides the dp4a(0x01010101,u) sum (sum_p (raw-4)u = dot_raw - 4*sum_u, like Q6_K's -32).
// The signed 6-bit sub-scale (scales[is]-32, per 16 elems) is the q3k_scales reconstruction. NSUB=8.
template <> struct qdp4a<DW_Q3_K> {
    static constexpr int NSUB = 8;
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int sub) {
        const int* hmask = reinterpret_cast<const int*>(blk);             // 32 B = 8 ints
        const int* qs    = reinterpret_cast<const int*>(blk + 32);        // 64 B = 16 ints
        int8_t sc[16];
        q3k_scales(blk + 96, sc);                                          // 12 bytes -> 16 i8
        const float d = __half2float(*reinterpret_cast<const __half*>(blk + 108));
        const int J = sub;                                                 // 32-block index 0..7
        const int shift = 2 * (J & 3);
        const int* qint = qs + (J >> 2) * 8;
        const int* u    = reinterpret_cast<const int*>(xq8) + J * 8;
        const float d8  = __half2float(xd8[J]);
        const int sc_lo = (int)sc[2 * J]     - 32;
        const int sc_hi = (int)sc[2 * J + 1] - 32;
        int dlo = 0, slo = 0, dhi = 0, shi = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {                                       // lo 16: ints 0..3
            const int raw = ((qint[i] >> shift) & 0x03030303) | (((hmask[i] >> J) & 0x01010101) << 2);
            dlo = hip_dp4a(raw, u[i], dlo);
            slo = hip_dp4a(0x01010101, u[i], slo);
        }
        #pragma unroll
        for (int i = 4; i < 8; ++i) {                                       // hi 16: ints 4..7
            const int raw = ((qint[i] >> shift) & 0x03030303) | (((hmask[i] >> J) & 0x01010101) << 2);
            dhi = hip_dp4a(raw, u[i], dhi);
            shi = hip_dp4a(0x01010101, u[i], shi);
        }
        return d * d8 * (float)(sc_lo * (dlo - 4 * slo) + sc_hi * (dhi - 4 * shi));
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int sub = 0; sub < NSUB; ++sub) s += partial(blk, xq8, xd8, sub);
        return s;
    }
};

// Per-type byte stride for the q8_1-paired dp4a path (= on-disk super-block bytes; mirrors
// qdw_traits::BYTES). All wired dp4a types are 256-element super-blocks.
template <int WTYPE> struct qdp4a_traits;
template <> struct qdp4a_traits<DW_Q4_K> { static constexpr int BYTES = 144; };
template <> struct qdp4a_traits<DW_Q6_K> { static constexpr int BYTES = 210; };
template <> struct qdp4a_traits<DW_Q2_K> { static constexpr int BYTES = 84;  };
template <> struct qdp4a_traits<DW_Q3_K> { static constexpr int BYTES = 110; };
template <> struct qdp4a_traits<DW_Q5_K> { static constexpr int BYTES = 176; };

// THE unified dp4a decode core. Warp per output row; lane-strided over the row's 256-elem
// super-blocks; per-type `qdp4a<WTYPE>::block` does the int8 dot of one super-block against its
// pre-quantized q8_1 activation slice; warp-shuffle reduces. XT is only the output store type (the
// q8_1 activation is already int8 + f16 scale, dtype-independent).
template <int WTYPE, typename XT>
__device__ __forceinline__ void qmatvec_dp4a_core(
    const int nrows,
    const int ncols,                  // multiple of 256
    const uint8_t* __restrict__ wq,
    const int8_t*  __restrict__ xq,   // [ncols] q8_1 int8 quants
    const __half*  __restrict__ xd,   // [ncols/32] q8_1 f16 scales
    XT* __restrict__ y
) {
    constexpr int WBYTES = qdp4a_traits<WTYPE>::BYTES;
    const int lane = threadIdx.x & 31;
    const int rows_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * rows_per_block + (threadIdx.x >> 5);
    if (row >= nrows) {
        return;
    }
    const int nblocks = ncols >> 8; // ncols / 256
    const uint8_t* row_ptr = wq + (size_t)row * (size_t)nblocks * WBYTES;

    // Stride lanes over (super-block x sub-unit) work items, NOT whole super-blocks -- the SAME fix the
    // MoE twin already carries. Narrow-k decode shapes (attn q/k/v/o k=2048 -> 8 super-blocks) left
    // 24/32 lanes idle under whole-super-block striding; NSUB sub-units (Q4_K=4 chunks, Q6_K=16 groups)
    // lift the work-item count to nblocks*NSUB so every lane carries a coalesced dp4a. block() = sum
    // over sub, so this is byte-identical math (f32 reorder only) -- the one lane-coalesced core.
    constexpr int NSUB = qdp4a<WTYPE>::NSUB;
    const int nunits = nblocks * NSUB;
    float acc = 0.0f;
    for (int u = lane; u < nunits; u += 32) {
        const int b = u / NSUB;
        const int sub = u - b * NSUB;
        acc += qdp4a<WTYPE>::partial(
            row_ptr + (size_t)b * WBYTES, xq + (size_t)b * 256, xd + (size_t)b * 8, sub);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off);
    }
    if (lane == 0) {
        qstore(&y[row], acc);
    }
}

// THE unified indexed-MoE dp4a decode core. Same per-row math as qmatvec_dp4a_core, but ALL routed
// slots run in ONE launch with experts on grid.y: slot s = blockIdx.y, expert = ids[s] (read
// ON-DEVICE), bank offset by ids[s] IN-KERNEL, output row `row` of that expert against slot s's
// pre-quantized q8_1 activation (xq + s*ncols, xd + s*ncols/32) into y[s*n + row]. Collapses the host
// per-expert launch loop into one well-occupied capture-clean grid + int8 dp4a.
template <int WTYPE, typename XT>
__device__ __forceinline__ void moe_qmatvec_dp4a_core(
    const int n,                        // output rows per expert (weight rows)
    const int ncols,                    // k, multiple of 256
    const int nslots,                   // routed slots (= nrows)
    const uint8_t* __restrict__ wbank,  // [E, n, k] blocks
    const int* __restrict__ ids,        // [nslots] expert id per slot
    const int8_t* __restrict__ xq,      // [nslots, ncols] q8_1 int8 quants
    const __half* __restrict__ xd,      // [nslots, ncols/32] q8_1 f16 scales
    XT* __restrict__ y                  // [nslots, n]
) {
    constexpr int WBYTES = qdp4a_traits<WTYPE>::BYTES;
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
    const uint8_t* row_ptr = wbank + ((size_t)expert * n + row) * (size_t)nblocks * WBYTES;
    const int8_t* xq_row = xq + (size_t)s * ncols;
    const __half* xd_row = xd + (size_t)s * (ncols >> 5);

    // Stride lanes over (super-block x sub-unit) work items, NOT whole super-blocks. The expert
    // shapes are narrow-k (gate/up k=2048 -> 8 super-blocks, down k=768 -> 3), so whole-super-block
    // striding leaves 24/32 (gate/up) or 29/32 (down) lanes idle. NSUB sub-units per super-block
    // (Q4_K=4 chunks, Q6_K=16 groups) lifts the parallel unit count to nblocks*NSUB (>=32 here), so
    // every lane carries dp4a work; sub-unit partials sum identically to qdp4a<WTYPE>::block.
    constexpr int NSUB = qdp4a<WTYPE>::NSUB;
    const int nunits = nblocks * NSUB;
    float acc = 0.0f;
    for (int u = lane; u < nunits; u += 32) {
        const int b = u / NSUB;
        const int sub = u - b * NSUB;
        acc += qdp4a<WTYPE>::partial(
            row_ptr + (size_t)b * WBYTES, xq_row + (size_t)b * 256, xd_row + (size_t)b * 8, sub);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off);
    }
    if (lane == 0) {
        qstore(&y[(size_t)s * n + row], acc);
    }
}

// Unified dp4a entry points (f16 + bf16, decode + MoE). Every symbol is the SAME core with a
// different WTYPE -- one dp4a core, one launcher table on the Rust side (dp4a_decode_kernel /
// dp4a_moe_kernel), selected by RocmQuantType::dp4a_capable. Adding a dp4a type = one qdp4a<WTYPE> +
// one qdp4a_traits<WTYPE> + one table row, NO new kernel. `qstore` is the shared type-dispatched
// store defined in the preamble above.
#define DEFINE_QMATVEC_DP4A(NAME, WTYPE)                                                             \
    extern "C" __global__ void qmatvec_dp4a_##NAME##_f16(                                            \
        const int nrows, const int ncols, const uint8_t* __restrict__ wq,                            \
        const int8_t* __restrict__ xq, const __half* __restrict__ xd, __half* __restrict__ y) {      \
        qmatvec_dp4a_core<WTYPE, __half>(nrows, ncols, wq, xq, xd, y);                                \
    }                                                                                                \
    extern "C" __global__ void qmatvec_dp4a_##NAME##_bf16(                                           \
        const int nrows, const int ncols, const uint8_t* __restrict__ wq,                            \
        const int8_t* __restrict__ xq, const __half* __restrict__ xd, hip_bfloat16* __restrict__ y) {\
        qmatvec_dp4a_core<WTYPE, hip_bfloat16>(nrows, ncols, wq, xq, xd, y);                          \
    }                                                                                                \
    extern "C" __global__ void moe_qmatvec_dp4a_##NAME##_f16(                                        \
        const int n, const int ncols, const int nslots, const uint8_t* __restrict__ wbank,           \
        const int* __restrict__ ids, const int8_t* __restrict__ xq, const __half* __restrict__ xd,   \
        __half* __restrict__ y) {                                                                    \
        moe_qmatvec_dp4a_core<WTYPE, __half>(n, ncols, nslots, wbank, ids, xq, xd, y);                \
    }                                                                                                \
    extern "C" __global__ void moe_qmatvec_dp4a_##NAME##_bf16(                                       \
        const int n, const int ncols, const int nslots, const uint8_t* __restrict__ wbank,           \
        const int* __restrict__ ids, const int8_t* __restrict__ xq, const __half* __restrict__ xd,   \
        hip_bfloat16* __restrict__ y) {                                                              \
        moe_qmatvec_dp4a_core<WTYPE, hip_bfloat16>(n, ncols, nslots, wbank, ids, xq, xd, y);          \
    }                                                                                                \
    extern "C" __global__ void qmatvec_dp4a_##NAME##_f32(                                            \
        const int nrows, const int ncols, const uint8_t* __restrict__ wq,                            \
        const int8_t* __restrict__ xq, const __half* __restrict__ xd, float* __restrict__ y) {       \
        qmatvec_dp4a_core<WTYPE, float>(nrows, ncols, wq, xq, xd, y);                                 \
    }                                                                                                \
    extern "C" __global__ void moe_qmatvec_dp4a_##NAME##_f32(                                        \
        const int n, const int ncols, const int nslots, const uint8_t* __restrict__ wbank,           \
        const int* __restrict__ ids, const int8_t* __restrict__ xq, const __half* __restrict__ xd,   \
        float* __restrict__ y) {                                                                     \
        moe_qmatvec_dp4a_core<WTYPE, float>(n, ncols, nslots, wbank, ids, xq, xd, y);                 \
    }

DEFINE_QMATVEC_DP4A(q4k, DW_Q4_K)
DEFINE_QMATVEC_DP4A(q6k, DW_Q6_K)
DEFINE_QMATVEC_DP4A(q2k, DW_Q2_K)
DEFINE_QMATVEC_DP4A(q3k, DW_Q3_K)
DEFINE_QMATVEC_DP4A(q5k, DW_Q5_K)
// NOTE: DEFINE_QMATVEC_DP4A is NOT #undef'd here -- the IQ codebook dp4a specializations below reuse
// the SAME macro (one launcher generator). They live after the __constant__ grid tables they index
// (IQ2XXS_GRID_D / IQ3XXS_GRID_D / IQ1S_GRID_D / KSIGNS_IQ2XS_D), so their qdp4a<> specializations +
// DEFINE rows + the #undef are emitted there, next to the scalar qdec<> IQ twins.

// Dense (non-quantized) GEMV for the decode rows==1 case: y[row] = sum_k w[row,k]*x[k]. Replaces the
// `broadcast_mul + sum` reduce that materialized the whole [n,k] product just to reduce it (e.g. the
// MoE F32 router gate [128,2048]: the broadcast_mul wrote + the sum re-read a 1MB intermediate every
// layer). One warp per output row, lanes strided over k, f32 accumulate, shuffle-reduce -- reads the
// weight row ONCE, no intermediate. Capture-clean (no rocBLAS). Same store-dispatch as the quant cores.
__device__ __forceinline__ float dgload(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float dgload(const float* p)  { return *p; }
// ONE block per output row, all DGBLK threads split the k-dot (n is small here -- the [128,2048]
// router gate -- so warp-per-row left only 16 blocks = latency-bound at ~13% BW). Block-per-row gives
// nrows blocks * DGBLK threads of memory-level parallelism + a fully coalesced row sweep (thread t
// reads w[row,t], t+DGBLK, ...), then a warp + shared reduce. Mirrors llama's RDNA nwarps>1 split-k.
#define DGBLK 256
template <typename T>
__device__ __forceinline__ void dense_gemv_core(
    const int nrows, const int ncols, const T* __restrict__ w, const T* __restrict__ x, T* __restrict__ y
) {
    const int row = blockIdx.x;
    if (row >= nrows) {
        return;
    }
    const int t = threadIdx.x;
    const T* wr = w + (size_t)row * (size_t)ncols;
    float acc = 0.0f;
    for (int j = t; j < ncols; j += DGBLK) {
        acc += dgload(wr + j) * dgload(x + j);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off);
    }
    __shared__ float warp_sums[DGBLK / 32];
    if ((t & 31) == 0) {
        warp_sums[t >> 5] = acc;
    }
    __syncthreads();
    if (t == 0) {
        float s = 0.0f;
        #pragma unroll
        for (int w_ = 0; w_ < DGBLK / 32; ++w_) {
            s += warp_sums[w_];
        }
        qstore(&y[row], s);
    }
}
extern "C" __global__ void dense_gemv_f16(
    const int nrows, const int ncols, const __half* __restrict__ w, const __half* __restrict__ x, __half* __restrict__ y) {
    dense_gemv_core<__half>(nrows, ncols, w, x, y);
}
extern "C" __global__ void dense_gemv_f32(
    const int nrows, const int ncols, const float* __restrict__ w, const float* __restrict__ x, float* __restrict__ y) {
    dense_gemv_core<float>(nrows, ncols, w, x, y);
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

// Byte k (0..11) of the 12 Q4_K scale bytes packed into 3 little-endian words (w0 = s[0..3],
// w1 = s[4..7], w2 = s[8..11]) -- the register-resident twin of `s[k]` so the dp4a path sources the
// scales from the one 128-bit header load, not byte gathers. Pure ALU (select + shift + mask).
__device__ __forceinline__ uint32_t q4k_sbyte(uint32_t w0, uint32_t w1, uint32_t w2, int k) {
    const uint32_t w = (k < 4) ? w0 : ((k < 8) ? w1 : w2);
    return (w >> (8 * (k & 3))) & 0xFFu;
}

// get_scale_min_k4 over the packed scale words -- BIT-IDENTICAL to q4k_scale_min(blk+4, j, ...): same
// bytes (via q4k_sbyte), same 6-bit arithmetic. Only the byte source is registers, not DRAM.
__device__ __forceinline__ void q4k_scale_min_words(uint32_t w0, uint32_t w1, uint32_t w2, int j, int* sc, int* m) {
    if (j < 4) {
        *sc = (int)(q4k_sbyte(w0, w1, w2, j)     & 63u);
        *m  = (int)(q4k_sbyte(w0, w1, w2, j + 4) & 63u);
    } else {
        const uint32_t a = q4k_sbyte(w0, w1, w2, j + 4);
        const uint32_t b = q4k_sbyte(w0, w1, w2, j - 4);
        const uint32_t c = q4k_sbyte(w0, w1, w2, j);
        *sc = (int)((a & 0xFu) | ((b >> 6) << 4));
        *m  = (int)((a >> 4)   | ((c >> 6) << 4));
    }
}

// q3k_scales (BlockQ3K::to_float scale unpack): the 12 packed scale bytes encode 16 6-bit sub-block
// scales -- the low 4 bits of each live in 2 u32, the high 2 bits are packed into the 3rd. Reconstructs
// 4 u32 from the 3 (the Rust `aux[]` transform bit-for-bit), then writes them out as 16 i8 (still
// biased: subtract 32 at use). Warp-uniform input (the 12 bytes are shared) so the ALU is amortized.
__device__ __forceinline__ void q3k_scales(const uint8_t* __restrict__ s, int8_t* __restrict__ out) {
    const uint32_t KMASK1 = 0x03030303u, KMASK2 = 0x0f0f0f0fu;
    uint32_t aux[4];
    aux[0] = (uint32_t)s[0] | ((uint32_t)s[1] << 8) | ((uint32_t)s[2] << 16) | ((uint32_t)s[3] << 24);
    aux[1] = (uint32_t)s[4] | ((uint32_t)s[5] << 8) | ((uint32_t)s[6] << 16) | ((uint32_t)s[7] << 24);
    aux[2] = (uint32_t)s[8] | ((uint32_t)s[9] << 8) | ((uint32_t)s[10] << 16) | ((uint32_t)s[11] << 24);
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | (((tmp) & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        out[g * 4 + 0] = (int8_t)(aux[g] & 0xFF);
        out[g * 4 + 1] = (int8_t)((aux[g] >> 8) & 0xFF);
        out[g * 4 + 2] = (int8_t)((aux[g] >> 16) & 0xFF);
        out[g * 4 + 3] = (int8_t)((aux[g] >> 24) & 0xFF);
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
// MXFP4 (gpt-oss) FP4 E2M1 codebook (iq_quants.rs KVALUES_MXFP4) + E8M0 1-byte exponent -> fp32
// scale, halved (ggml-impl.h ggml_e8m0_to_fp32_half). qdec<DW_MXFP4> mirrors qdec<DW_IQ4_NL>: the
// scale is the E8M0 byte (not an f16), and qs starts at blk+1 (1-byte scale, not 2).
__device__ __constant__ int8_t KVALUES_MXFP4_D[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};
__device__ __forceinline__ float e8m0_to_fp32_half(uint8_t x) {
    const uint32_t bits = (x < 2) ? (0x00200000u << x) : ((uint32_t)(x - 1) << 23);
    return __uint_as_float(bits);
}
// IQ1_S/IQ1_M delta bias (IQ1S_DELTA, iq_quants.rs:28). dequant adds +/-delta to the signed
// grid byte before scaling: val = dl*(grid_i8 + delta). delta sign is a per-group qh bit.
constexpr float IQ1S_DELTA = 0.125f;
// IQ1_S/IQ1_M shared 11-bit signed-grid codebook (IQ1S_GRID, iq_grids.rs:342). Each uint64 packs
// 8 int8 coords in {-1,0,+1} (bytes 0xff/0x00/0x01). Generated byte-identical from the Rust table;
// the qmatvec_iq1_numeric oracle (GPU decode == CPU to_float, nbad=0) is the bit-exactness gate.
__device__ __constant__ uint64_t IQ1S_GRID_D[2048] = {
    0xffffffffffffffffULL, 0xffffffffffffff01ULL, 0xffffffffffff0000ULL, 0xffffffffffff01ffULL, 0xffffffffffff0101ULL, 0xffffffffff00ff00ULL, 0xffffffffff000000ULL, 0xffffffffff01ffffULL,
    0xffffffffff01ff01ULL, 0xffffffffff0101ffULL, 0xffffffffff010101ULL, 0xffffffff00ff0000ULL, 0xffffffff0000ff00ULL, 0xffffffff000000ffULL, 0xffffffff00000001ULL, 0xffffffff00010000ULL,
    0xffffffff01ffffffULL, 0xffffffff01ffff01ULL, 0xffffffff01ff01ffULL, 0xffffffff01ff0101ULL, 0xffffffff01000000ULL, 0xffffffff0101ffffULL, 0xffffffff0101ff01ULL, 0xffffffff010101ffULL,
    0xffffffff01010101ULL, 0xffffff00ffff00ffULL, 0xffffff00ffff0000ULL, 0xffffff00ff00ff00ULL, 0xffffff00ff0000ffULL, 0xffffff00ff000001ULL, 0xffffff00ff000100ULL, 0xffffff00ff000101ULL,
    0xffffff00ff010000ULL, 0xffffff0000ffff00ULL, 0xffffff0000ff0001ULL, 0xffffff0000ff0100ULL, 0xffffff000000ff01ULL, 0xffffff0000000000ULL, 0xffffff0000000101ULL, 0xffffff000001ff00ULL,
    0xffffff00000100ffULL, 0xffffff0000010001ULL, 0xffffff00000101ffULL, 0xffffff0001ff0000ULL, 0xffffff000100ff00ULL, 0xffffff00010000ffULL, 0xffffff0001000001ULL, 0xffffff0001010000ULL,
    0xffffff01ffffffffULL, 0xffffff01ffffff01ULL, 0xffffff01ffff01ffULL, 0xffffff01ffff0101ULL, 0xffffff01ff000000ULL, 0xffffff01ff01ffffULL, 0xffffff01ff01ff01ULL, 0xffffff01ff0101ffULL,
    0xffffff01ff010101ULL, 0xffffff0100ff0000ULL, 0xffffff010000ff00ULL, 0xffffff0100000100ULL, 0xffffff01000100ffULL, 0xffffff0100010100ULL, 0xffffff0101ffffffULL, 0xffffff0101ffff01ULL,
    0xffffff0101ff01ffULL, 0xffffff0101ff0101ULL, 0xffffff010100ff00ULL, 0xffffff0101000000ULL, 0xffffff0101000100ULL, 0xffffff010101ffffULL, 0xffffff010101ff01ULL, 0xffffff01010101ffULL,
    0xffffff0101010101ULL, 0xffff00ffff00ff00ULL, 0xffff00ffff0000ffULL, 0xffff00ffff000001ULL, 0xffff00ffff010000ULL, 0xffff00ff00ffff00ULL, 0xffff00ff00ff0100ULL, 0xffff00ff00000000ULL,
    0xffff00ff00000101ULL, 0xffff00ff000100ffULL, 0xffff00ff00010000ULL, 0xffff00ff0100ff00ULL, 0xffff00ff01000100ULL, 0xffff00ff01010000ULL, 0xffff0000ffffff00ULL, 0xffff0000ffff00ffULL,
    0xffff0000ffff0000ULL, 0xffff0000ffff0001ULL, 0xffff0000ff000000ULL, 0xffff0000ff0001ffULL, 0xffff0000ff000101ULL, 0xffff0000ff010100ULL, 0xffff000000ffffffULL, 0xffff000000ff0000ULL,
    0xffff000000ff0101ULL, 0xffff00000000ffffULL, 0xffff00000000ff00ULL, 0xffff0000000000ffULL, 0xffff000000000000ULL, 0xffff000000000001ULL, 0xffff000000000100ULL, 0xffff00000001ffffULL,
    0xffff00000001ff01ULL, 0xffff000000010000ULL, 0xffff0000000101ffULL, 0xffff000000010101ULL, 0xffff000001ffff00ULL, 0xffff00000100ff00ULL, 0xffff000001000000ULL, 0xffff0000010001ffULL,
    0xffff000001000101ULL, 0xffff00000101ff00ULL, 0xffff0000010100ffULL, 0xffff000001010000ULL, 0xffff000001010001ULL, 0xffff000001010100ULL, 0xffff0001ff0000ffULL, 0xffff0001ff000100ULL,
    0xffff000100ffff00ULL, 0xffff000100ff00ffULL, 0xffff00010000ffffULL, 0xffff00010000ff01ULL, 0xffff000100000000ULL, 0xffff0001000001ffULL, 0xffff00010001ffffULL, 0xffff00010001ff00ULL,
    0xffff000100010001ULL, 0xffff000100010100ULL, 0xffff000101ff0000ULL, 0xffff00010100ff00ULL, 0xffff0001010000ffULL, 0xffff000101000100ULL, 0xffff01ffffffffffULL, 0xffff01ffffffff01ULL,
    0xffff01ffffff01ffULL, 0xffff01ffffff0101ULL, 0xffff01ffff000000ULL, 0xffff01ffff01ffffULL, 0xffff01ffff01ff01ULL, 0xffff01ffff0101ffULL, 0xffff01ffff010101ULL, 0xffff01ff00ff0000ULL,
    0xffff01ff0000ff00ULL, 0xffff01ff00000001ULL, 0xffff01ff00010000ULL, 0xffff01ff01ffffffULL, 0xffff01ff01ffff01ULL, 0xffff01ff01ff01ffULL, 0xffff01ff01ff0101ULL, 0xffff01ff01000000ULL,
    0xffff01ff0101ffffULL, 0xffff01ff0101ff01ULL, 0xffff01ff010101ffULL, 0xffff01ff01010101ULL, 0xffff0100ffff0000ULL, 0xffff0100ff00ff00ULL, 0xffff0100ff0000ffULL, 0xffff0100ff000100ULL,
    0xffff0100ff0100ffULL, 0xffff0100ff010000ULL, 0xffff010000ffff00ULL, 0xffff01000000ffffULL, 0xffff01000000ff00ULL, 0xffff010000000000ULL, 0xffff01000001ff00ULL, 0xffff0100000100ffULL,
    0xffff010000010100ULL, 0xffff01000100ff00ULL, 0xffff0100010000ffULL, 0xffff010001000001ULL, 0xffff010001000100ULL, 0xffff010001010000ULL, 0xffff0101ffffffffULL, 0xffff0101ffffff01ULL,
    0xffff0101ffff01ffULL, 0xffff0101ffff0101ULL, 0xffff0101ff000000ULL, 0xffff0101ff01ffffULL, 0xffff0101ff01ff01ULL, 0xffff0101ff0101ffULL, 0xffff0101ff010101ULL, 0xffff010100ff0000ULL,
    0xffff01010000ff00ULL, 0xffff010100000100ULL, 0xffff01010001ff00ULL, 0xffff010100010000ULL, 0xffff010101ffffffULL, 0xffff010101ffff01ULL, 0xffff010101ff0000ULL, 0xffff010101ff01ffULL,
    0xffff010101ff0101ULL, 0xffff010101000000ULL, 0xffff01010101ffffULL, 0xffff01010101ff01ULL, 0xffff0101010101ffULL, 0xffff010101010101ULL, 0xff00ffffff00ffffULL, 0xff00ffffff00ff00ULL,
    0xff00ffffff0000ffULL, 0xff00ffffff000100ULL, 0xff00ffffff0100ffULL, 0xff00ffffff010000ULL, 0xff00ffff00ffff00ULL, 0xff00ffff00ff00ffULL, 0xff00ffff0000ffffULL, 0xff00ffff00000000ULL,
    0xff00ffff000001ffULL, 0xff00ffff0001ff00ULL, 0xff00ffff000100ffULL, 0xff00ffff00010000ULL, 0xff00ffff00010100ULL, 0xff00ffff0100ff00ULL, 0xff00ffff010000ffULL, 0xff00ffff01000001ULL,
    0xff00ffff0101ff00ULL, 0xff00ffff01010000ULL, 0xff00ff00ffffff00ULL, 0xff00ff00ffff00ffULL, 0xff00ff00ffff0001ULL, 0xff00ff00ffff0100ULL, 0xff00ff00ff00ffffULL, 0xff00ff00ff00ff01ULL,
    0xff00ff00ff000000ULL, 0xff00ff00ff0001ffULL, 0xff00ff00ff01ff00ULL, 0xff00ff00ff0100ffULL, 0xff00ff00ff010100ULL, 0xff00ff0000ff0000ULL, 0xff00ff0000ff0101ULL, 0xff00ff000000ffffULL,
    0xff00ff000000ff00ULL, 0xff00ff000000ff01ULL, 0xff00ff00000000ffULL, 0xff00ff0000000000ULL, 0xff00ff0000000001ULL, 0xff00ff0000000100ULL, 0xff00ff000001ffffULL, 0xff00ff0000010000ULL,
    0xff00ff0001ff00ffULL, 0xff00ff000100ff01ULL, 0xff00ff0001000000ULL, 0xff00ff000101ff00ULL, 0xff00ff00010100ffULL, 0xff00ff01ff00ff00ULL, 0xff00ff01ff0000ffULL, 0xff00ff01ff000001ULL,
    0xff00ff01ff010000ULL, 0xff00ff0100ffffffULL, 0xff00ff0100ff0001ULL, 0xff00ff0100ff0100ULL, 0xff00ff010000ff01ULL, 0xff00ff0100000000ULL, 0xff00ff01000001ffULL, 0xff00ff0100000101ULL,
    0xff00ff01000100ffULL, 0xff00ff0100010001ULL, 0xff00ff0101ff0000ULL, 0xff00ff010100ff00ULL, 0xff00ff01010000ffULL, 0xff00ff0101000001ULL, 0xff00ff0101010000ULL, 0xff0000ffffffff00ULL,
    0xff0000ffffff0001ULL, 0xff0000ffffff0100ULL, 0xff0000ffff0000ffULL, 0xff0000ffff000000ULL, 0xff0000ffff0001ffULL, 0xff0000ffff000100ULL, 0xff0000ffff01ff00ULL, 0xff0000ffff010001ULL,
    0xff0000ff00ffff00ULL, 0xff0000ff00ff0000ULL, 0xff0000ff00ff0001ULL, 0xff0000ff00ff01ffULL, 0xff0000ff00ff0101ULL, 0xff0000ff0000ff00ULL, 0xff0000ff000000ffULL, 0xff0000ff00000000ULL,
    0xff0000ff00000001ULL, 0xff0000ff00000100ULL, 0xff0000ff0001ff01ULL, 0xff0000ff00010000ULL, 0xff0000ff000101ffULL, 0xff0000ff01ff00ffULL, 0xff0000ff01ff0100ULL, 0xff0000ff0100ffffULL,
    0xff0000ff010000ffULL, 0xff0000ff01000000ULL, 0xff0000ff010001ffULL, 0xff0000ff01000100ULL, 0xff0000ff01000101ULL, 0xff0000ff0101ff00ULL, 0xff0000ff010100ffULL, 0xff0000ff01010000ULL,
    0xff0000ff01010100ULL, 0xff000000ffffff01ULL, 0xff000000ffff0000ULL, 0xff000000ffff0101ULL, 0xff000000ff00ff00ULL, 0xff000000ff0000ffULL, 0xff000000ff000000ULL, 0xff000000ff000001ULL,
    0xff000000ff000100ULL, 0xff000000ff01ffffULL, 0xff000000ff01ff01ULL, 0xff000000ff010000ULL, 0xff000000ff0101ffULL, 0xff000000ff010101ULL, 0xff00000000ffff00ULL, 0xff00000000ff00ffULL,
    0xff00000000ff0000ULL, 0xff00000000ff0001ULL, 0xff0000000000ff00ULL, 0xff0000000000ff01ULL, 0xff000000000000ffULL, 0xff00000000000000ULL, 0xff00000000000001ULL, 0xff00000000000100ULL,
    0xff00000000000101ULL, 0xff0000000001ff00ULL, 0xff000000000100ffULL, 0xff00000000010000ULL, 0xff00000000010001ULL, 0xff00000000010100ULL, 0xff00000001ffffffULL, 0xff00000001ffff01ULL,
    0xff00000001ff00ffULL, 0xff00000001ff0000ULL, 0xff00000001ff01ffULL, 0xff00000001ff0101ULL, 0xff0000000100ffffULL, 0xff0000000100ff00ULL, 0xff000000010000ffULL, 0xff00000001000000ULL,
    0xff00000001000001ULL, 0xff00000001000100ULL, 0xff00000001000101ULL, 0xff0000000101ffffULL, 0xff0000000101ff01ULL, 0xff00000001010000ULL, 0xff000001ffffff00ULL, 0xff000001ffff00ffULL,
    0xff000001ffff0000ULL, 0xff000001ffff0001ULL, 0xff000001ff000000ULL, 0xff000001ff000001ULL, 0xff000001ff0001ffULL, 0xff000001ff000101ULL, 0xff000001ff01ff00ULL, 0xff000001ff010001ULL,
    0xff00000100ffffffULL, 0xff00000100ffff01ULL, 0xff00000100ff00ffULL, 0xff00000100ff0000ULL, 0xff00000100ff01ffULL, 0xff00000100ff0101ULL, 0xff0000010000ff00ULL, 0xff00000100000000ULL,
    0xff00000100000001ULL, 0xff000001000001ffULL, 0xff00000100000100ULL, 0xff0000010001ff00ULL, 0xff000001000100ffULL, 0xff00000100010000ULL, 0xff000001000101ffULL, 0xff00000100010100ULL,
    0xff00000100010101ULL, 0xff00000101ff0001ULL, 0xff00000101ff0101ULL, 0xff0000010100ff01ULL, 0xff00000101000000ULL, 0xff000001010100ffULL, 0xff00000101010100ULL, 0xff0001ffff00ff00ULL,
    0xff0001ffff000001ULL, 0xff0001ffff010000ULL, 0xff0001ff00ffff00ULL, 0xff0001ff00ff00ffULL, 0xff0001ff00ff0001ULL, 0xff0001ff00ff0100ULL, 0xff0001ff0000ffffULL, 0xff0001ff00000000ULL,
    0xff0001ff000001ffULL, 0xff0001ff00000101ULL, 0xff0001ff0001ffffULL, 0xff0001ff0001ff00ULL, 0xff0001ff000100ffULL, 0xff0001ff00010001ULL, 0xff0001ff00010100ULL, 0xff0001ff01ff0000ULL,
    0xff0001ff0100ff00ULL, 0xff0001ff010000ffULL, 0xff0001ff01010000ULL, 0xff000100ff00ffffULL, 0xff000100ff00ff01ULL, 0xff000100ff000000ULL, 0xff000100ff000101ULL, 0xff000100ff01ff00ULL,
    0xff000100ff010000ULL, 0xff00010000ffff01ULL, 0xff00010000ff00ffULL, 0xff00010000ff0000ULL, 0xff00010000ff01ffULL, 0xff0001000000ff00ULL, 0xff000100000000ffULL, 0xff00010000000000ULL,
    0xff00010000000001ULL, 0xff00010000000100ULL, 0xff00010000000101ULL, 0xff0001000001ffffULL, 0xff00010000010000ULL, 0xff00010000010101ULL, 0xff00010001ff0100ULL, 0xff0001000100ff00ULL,
    0xff0001000100ff01ULL, 0xff00010001000000ULL, 0xff000100010001ffULL, 0xff0001000101ff00ULL, 0xff00010001010001ULL, 0xff00010001010100ULL, 0xff000101ffff0100ULL, 0xff000101ff000001ULL,
    0xff000101ff0100ffULL, 0xff000101ff010001ULL, 0xff00010100ff00ffULL, 0xff00010100ff0001ULL, 0xff00010100ff0100ULL, 0xff0001010000ffffULL, 0xff0001010000ff01ULL, 0xff00010100000000ULL,
    0xff000101000001ffULL, 0xff0001010001ff00ULL, 0xff00010100010001ULL, 0xff00010100010100ULL, 0xff00010101ff0000ULL, 0xff0001010100ff00ULL, 0xff00010101000001ULL, 0xff00010101000101ULL,
    0xff01ffffffffffffULL, 0xff01ffffffffff01ULL, 0xff01ffffffff01ffULL, 0xff01ffffffff0101ULL, 0xff01ffffff000000ULL, 0xff01ffffff01ffffULL, 0xff01ffffff01ff01ULL, 0xff01ffffff010000ULL,
    0xff01ffffff0101ffULL, 0xff01ffffff010101ULL, 0xff01ffff00ff0000ULL, 0xff01ffff0000ff00ULL, 0xff01ffff00000100ULL, 0xff01ffff0001ff00ULL, 0xff01ffff00010000ULL, 0xff01ffff01ffffffULL,
    0xff01ffff01ffff01ULL, 0xff01ffff01ff01ffULL, 0xff01ffff01ff0101ULL, 0xff01ffff01000000ULL, 0xff01ffff0101ffffULL, 0xff01ffff0101ff01ULL, 0xff01ffff01010000ULL, 0xff01ffff010101ffULL,
    0xff01ffff01010101ULL, 0xff01ff00ffff0000ULL, 0xff01ff00ff00ff00ULL, 0xff01ff00ff0000ffULL, 0xff01ff00ff000100ULL, 0xff01ff00ff010000ULL, 0xff01ff0000ffff01ULL, 0xff01ff0000ff00ffULL,
    0xff01ff0000ff0100ULL, 0xff01ff0000000000ULL, 0xff01ff00000001ffULL, 0xff01ff0000000101ULL, 0xff01ff000001ff00ULL, 0xff01ff00000100ffULL, 0xff01ff0000010000ULL, 0xff01ff0000010001ULL,
    0xff01ff0001ff0000ULL, 0xff01ff000100ffffULL, 0xff01ff0001000001ULL, 0xff01ff0001000100ULL, 0xff01ff0001010000ULL, 0xff01ff01ffffff00ULL, 0xff01ff01ffff01ffULL, 0xff01ff01ffff0101ULL,
    0xff01ff01ff00ff00ULL, 0xff01ff01ff000000ULL, 0xff01ff01ff01ffffULL, 0xff01ff01ff01ff01ULL, 0xff01ff01ff0101ffULL, 0xff01ff01ff010101ULL, 0xff01ff0100ff0000ULL, 0xff01ff010000ff00ULL,
    0xff01ff0100000001ULL, 0xff01ff0100000100ULL, 0xff01ff0100010000ULL, 0xff01ff0101ffff00ULL, 0xff01ff0101ff01ffULL, 0xff01ff0101ff0101ULL, 0xff01ff010100ff00ULL, 0xff01ff0101000000ULL,
    0xff01ff010101ffffULL, 0xff01ff010101ff01ULL, 0xff01ff01010101ffULL, 0xff01ff0101010101ULL, 0xff0100ffffff0000ULL, 0xff0100ffff0000ffULL, 0xff0100ffff000001ULL, 0xff0100ffff000100ULL,
    0xff0100ffff010000ULL, 0xff0100ff00ff00ffULL, 0xff0100ff00ff0000ULL, 0xff0100ff00ff0001ULL, 0xff0100ff00ff0100ULL, 0xff0100ff0000ff01ULL, 0xff0100ff00000000ULL, 0xff0100ff000001ffULL,
    0xff0100ff00000101ULL, 0xff0100ff00010001ULL, 0xff0100ff01ff0000ULL, 0xff0100ff0100ff00ULL, 0xff0100ff010000ffULL, 0xff0100ff01000100ULL, 0xff0100ff0101ff00ULL, 0xff0100ff01010000ULL,
    0xff010000ffff0100ULL, 0xff010000ff000000ULL, 0xff010000ff01ff00ULL, 0xff010000ff010100ULL, 0xff01000000ffffffULL, 0xff01000000ff0000ULL, 0xff01000000ff01ffULL, 0xff0100000000ff00ULL,
    0xff010000000000ffULL, 0xff01000000000000ULL, 0xff01000000000100ULL, 0xff0100000001ff01ULL, 0xff01000000010000ULL, 0xff010000000101ffULL, 0xff01000001ff0100ULL, 0xff0100000100ffffULL,
    0xff010000010000ffULL, 0xff01000001000000ULL, 0xff010000010001ffULL, 0xff01000001000101ULL, 0xff0100000101ff00ULL, 0xff010000010100ffULL, 0xff01000001010001ULL, 0xff01000001010100ULL,
    0xff010001ffff0000ULL, 0xff010001ff00ffffULL, 0xff010001ff00ff01ULL, 0xff010001ff000100ULL, 0xff010001ff010000ULL, 0xff01000100ffff00ULL, 0xff01000100ff0100ULL, 0xff01000100000000ULL,
    0xff0100010001ffffULL, 0xff0100010001ff00ULL, 0xff01000100010100ULL, 0xff01000101ff00ffULL, 0xff01000101ff0001ULL, 0xff0100010100ffffULL, 0xff01000101000101ULL, 0xff0101ffffffffffULL,
    0xff0101ffffffff01ULL, 0xff0101ffffff01ffULL, 0xff0101ffffff0101ULL, 0xff0101ffff000000ULL, 0xff0101ffff01ffffULL, 0xff0101ffff01ff01ULL, 0xff0101ffff0101ffULL, 0xff0101ffff010101ULL,
    0xff0101ff00ff0000ULL, 0xff0101ff0000ff00ULL, 0xff0101ff000000ffULL, 0xff0101ff00010000ULL, 0xff0101ff01ffffffULL, 0xff0101ff01ffff01ULL, 0xff0101ff01ff01ffULL, 0xff0101ff01ff0101ULL,
    0xff0101ff0101ffffULL, 0xff0101ff0101ff01ULL, 0xff0101ff010101ffULL, 0xff0101ff01010101ULL, 0xff010100ffff0100ULL, 0xff010100ff00ff00ULL, 0xff010100ff0000ffULL, 0xff010100ff000100ULL,
    0xff010100ff010000ULL, 0xff01010000ff0001ULL, 0xff01010000ff0100ULL, 0xff0101000000ff01ULL, 0xff01010000000000ULL, 0xff0101000001ff00ULL, 0xff010100000100ffULL, 0xff01010000010001ULL,
    0xff01010000010100ULL, 0xff01010001ff0000ULL, 0xff0101000100ffffULL, 0xff01010001000001ULL, 0xff01010001000100ULL, 0xff010100010100ffULL, 0xff01010001010000ULL, 0xff010101ffffffffULL,
    0xff010101ffffff01ULL, 0xff010101ffff01ffULL, 0xff010101ffff0101ULL, 0xff010101ff01ffffULL, 0xff010101ff01ff01ULL, 0xff010101ff0101ffULL, 0xff010101ff010101ULL, 0xff01010100ff0000ULL,
    0xff0101010000ff00ULL, 0xff01010100000001ULL, 0xff01010100000100ULL, 0xff01010100010000ULL, 0xff01010101ffffffULL, 0xff01010101ffff01ULL, 0xff01010101ff01ffULL, 0xff01010101ff0101ULL,
    0xff01010101000000ULL, 0xff0101010101ffffULL, 0xff0101010101ff01ULL, 0xff010101010101ffULL, 0xff01010101010101ULL, 0x00ffffffffff0000ULL, 0x00ffffffff00ff00ULL, 0x00ffffffff000001ULL,
    0x00ffffffff010000ULL, 0x00ffffff00ff0100ULL, 0x00ffffff0000ff01ULL, 0x00ffffff00000000ULL, 0x00ffffff000001ffULL, 0x00ffffff00000101ULL, 0x00ffffff0001ff00ULL, 0x00ffffff000100ffULL,
    0x00ffffff00010001ULL, 0x00ffffff010000ffULL, 0x00ffffff01000100ULL, 0x00ffffff0101ff00ULL, 0x00ffffff01010001ULL, 0x00ffff00ffffffffULL, 0x00ffff00ffffff00ULL, 0x00ffff00ffff00ffULL,
    0x00ffff00ffff0001ULL, 0x00ffff00ffff0100ULL, 0x00ffff00ff00ff01ULL, 0x00ffff00ff000000ULL, 0x00ffff00ff000001ULL, 0x00ffff00ff0001ffULL, 0x00ffff00ff000101ULL, 0x00ffff00ff01ff00ULL,
    0x00ffff00ff010001ULL, 0x00ffff00ff010100ULL, 0x00ffff0000ff0000ULL, 0x00ffff0000ff01ffULL, 0x00ffff0000ff0101ULL, 0x00ffff000000ff00ULL, 0x00ffff00000000ffULL, 0x00ffff0000000000ULL,
    0x00ffff0000000001ULL, 0x00ffff0000000100ULL, 0x00ffff0000000101ULL, 0x00ffff0000010000ULL, 0x00ffff00000101ffULL, 0x00ffff0000010101ULL, 0x00ffff0001ffff00ULL, 0x00ffff0001ff00ffULL,
    0x00ffff0001ff0001ULL, 0x00ffff000100ffffULL, 0x00ffff000100ff01ULL, 0x00ffff0001000000ULL, 0x00ffff000101ffffULL, 0x00ffff000101ff00ULL, 0x00ffff000101ff01ULL, 0x00ffff01ffff0000ULL,
    0x00ffff01ff00ff00ULL, 0x00ffff01ff0000ffULL, 0x00ffff01ff000001ULL, 0x00ffff01ff010000ULL, 0x00ffff0100ffff00ULL, 0x00ffff010000ff01ULL, 0x00ffff0100000000ULL, 0x00ffff0100000101ULL,
    0x00ffff01000100ffULL, 0x00ffff0100010100ULL, 0x00ffff0101ff0100ULL, 0x00ffff01010000ffULL, 0x00ffff0101010000ULL, 0x00ff00ffffffff00ULL, 0x00ff00ffff000000ULL, 0x00ff00ffff000100ULL,
    0x00ff00ffff010100ULL, 0x00ff00ff00ff0000ULL, 0x00ff00ff00ff01ffULL, 0x00ff00ff00ff0101ULL, 0x00ff00ff0000ff00ULL, 0x00ff00ff000000ffULL, 0x00ff00ff00000000ULL, 0x00ff00ff00000001ULL,
    0x00ff00ff0001ff00ULL, 0x00ff00ff0001ff01ULL, 0x00ff00ff00010000ULL, 0x00ff00ff000101ffULL, 0x00ff00ff00010101ULL, 0x00ff00ff01ffff00ULL, 0x00ff00ff01ff0001ULL, 0x00ff00ff01ff0100ULL,
    0x00ff00ff0100ffffULL, 0x00ff00ff0100ff01ULL, 0x00ff00ff01000000ULL, 0x00ff00ff0101ffffULL, 0x00ff00ff0101ff00ULL, 0x00ff00ff01010100ULL, 0x00ff0000ffffff00ULL, 0x00ff0000ffffff01ULL,
    0x00ff0000ffff0000ULL, 0x00ff0000ffff0101ULL, 0x00ff0000ff00ff00ULL, 0x00ff0000ff0000ffULL, 0x00ff0000ff000000ULL, 0x00ff0000ff000001ULL, 0x00ff0000ff000100ULL, 0x00ff0000ff01ffffULL,
    0x00ff0000ff010000ULL, 0x00ff0000ff010101ULL, 0x00ff000000ffff00ULL, 0x00ff000000ff00ffULL, 0x00ff000000ff0000ULL, 0x00ff000000ff0001ULL, 0x00ff000000ff0100ULL, 0x00ff00000000ffffULL,
    0x00ff00000000ff00ULL, 0x00ff0000000000ffULL, 0x00ff000000000000ULL, 0x00ff000000000001ULL, 0x00ff0000000001ffULL, 0x00ff000000000100ULL, 0x00ff00000001ff00ULL, 0x00ff0000000100ffULL,
    0x00ff000000010000ULL, 0x00ff000000010001ULL, 0x00ff000000010100ULL, 0x00ff000001ffff01ULL, 0x00ff000001ff00ffULL, 0x00ff000001ff0000ULL, 0x00ff000001ff01ffULL, 0x00ff00000100ff00ULL,
    0x00ff0000010000ffULL, 0x00ff000001000000ULL, 0x00ff000001000001ULL, 0x00ff000001000100ULL, 0x00ff000001000101ULL, 0x00ff000001010000ULL, 0x00ff0000010101ffULL, 0x00ff000001010101ULL,
    0x00ff0001ffffff00ULL, 0x00ff0001ffff0000ULL, 0x00ff0001ffff0100ULL, 0x00ff0001ff0000ffULL, 0x00ff0001ff000000ULL, 0x00ff0001ff0001ffULL, 0x00ff0001ff000101ULL, 0x00ff0001ff01ff00ULL,
    0x00ff0001ff0100ffULL, 0x00ff0001ff010100ULL, 0x00ff000100ffffffULL, 0x00ff000100ffff01ULL, 0x00ff000100ff0000ULL, 0x00ff000100ff01ffULL, 0x00ff00010000ffffULL, 0x00ff00010000ff00ULL,
    0x00ff00010000ff01ULL, 0x00ff000100000000ULL, 0x00ff000100000001ULL, 0x00ff000100000100ULL, 0x00ff00010001ff01ULL, 0x00ff000100010000ULL, 0x00ff0001000101ffULL, 0x00ff000101ffff00ULL,
    0x00ff000101ff0000ULL, 0x00ff000101ff0101ULL, 0x00ff0001010000ffULL, 0x00ff000101000000ULL, 0x00ff00010101ff00ULL, 0x00ff0001010100ffULL, 0x00ff000101010001ULL, 0x00ff01ffffff0000ULL,
    0x00ff01ffff00ff00ULL, 0x00ff01ffff000000ULL, 0x00ff01ffff000101ULL, 0x00ff01ffff010000ULL, 0x00ff01ff00ffff01ULL, 0x00ff01ff00ff0100ULL, 0x00ff01ff0000ffffULL, 0x00ff01ff00000000ULL,
    0x00ff01ff000001ffULL, 0x00ff01ff0001ff00ULL, 0x00ff01ff000100ffULL, 0x00ff01ff00010001ULL, 0x00ff01ff00010100ULL, 0x00ff01ff01ff0000ULL, 0x00ff01ff0100ff00ULL, 0x00ff01ff010000ffULL,
    0x00ff01ff01000001ULL, 0x00ff01ff01000100ULL, 0x00ff01ff01010000ULL, 0x00ff0100ffffff00ULL, 0x00ff0100ffff0000ULL, 0x00ff0100ffff0001ULL, 0x00ff0100ffff0101ULL, 0x00ff0100ff00ffffULL,
    0x00ff0100ff0000ffULL, 0x00ff0100ff000000ULL, 0x00ff0100ff0001ffULL, 0x00ff0100ff01ff00ULL, 0x00ff0100ff0100ffULL, 0x00ff0100ff010001ULL, 0x00ff010000ffffffULL, 0x00ff010000ff0000ULL,
    0x00ff010000ff0101ULL, 0x00ff01000000ff00ULL, 0x00ff01000000ff01ULL, 0x00ff0100000000ffULL, 0x00ff010000000000ULL, 0x00ff010000000001ULL, 0x00ff010000000100ULL, 0x00ff01000001ffffULL,
    0x00ff01000001ff01ULL, 0x00ff010000010000ULL, 0x00ff010000010001ULL, 0x00ff010000010101ULL, 0x00ff010001ff0001ULL, 0x00ff010001ff0100ULL, 0x00ff01000100ff01ULL, 0x00ff010001000000ULL,
    0x00ff010001000001ULL, 0x00ff0100010001ffULL, 0x00ff01000101ff00ULL, 0x00ff0100010100ffULL, 0x00ff010001010001ULL, 0x00ff010001010100ULL, 0x00ff0101ff000001ULL, 0x00ff010100ff00ffULL,
    0x00ff010100ff0001ULL, 0x00ff010100ff0100ULL, 0x00ff010100000000ULL, 0x00ff0101000001ffULL, 0x00ff010100000101ULL, 0x00ff0101000100ffULL, 0x00ff010100010100ULL, 0x00ff0101010000ffULL,
    0x00ff010101010000ULL, 0x0000ffffffffff00ULL, 0x0000ffffffff00ffULL, 0x0000ffffffff0000ULL, 0x0000ffffffff0001ULL, 0x0000ffffffff0100ULL, 0x0000ffffff00ff01ULL, 0x0000ffffff000000ULL,
    0x0000ffffff000101ULL, 0x0000ffffff01ff00ULL, 0x0000ffffff0100ffULL, 0x0000ffffff010100ULL, 0x0000ffff00ffffffULL, 0x0000ffff00ff0000ULL, 0x0000ffff00ff01ffULL, 0x0000ffff0000ff00ULL,
    0x0000ffff000000ffULL, 0x0000ffff00000000ULL, 0x0000ffff00000001ULL, 0x0000ffff00000100ULL, 0x0000ffff00010000ULL, 0x0000ffff000101ffULL, 0x0000ffff01ff0001ULL, 0x0000ffff01ff0100ULL,
    0x0000ffff01000000ULL, 0x0000ffff010001ffULL, 0x0000ffff0101ffffULL, 0x0000ffff0101ff00ULL, 0x0000ffff01010001ULL, 0x0000ffff01010100ULL, 0x0000ff00ffff0000ULL, 0x0000ff00ffff01ffULL,
    0x0000ff00ffff0100ULL, 0x0000ff00ffff0101ULL, 0x0000ff00ff00ff00ULL, 0x0000ff00ff0000ffULL, 0x0000ff00ff000000ULL, 0x0000ff00ff000001ULL, 0x0000ff00ff0001ffULL, 0x0000ff00ff000100ULL,
    0x0000ff00ff01ffffULL, 0x0000ff00ff010000ULL, 0x0000ff00ff010001ULL, 0x0000ff00ff0101ffULL, 0x0000ff00ff010101ULL, 0x0000ff0000ffff00ULL, 0x0000ff0000ff00ffULL, 0x0000ff0000ff0000ULL,
    0x0000ff0000ff0001ULL, 0x0000ff0000ff0100ULL, 0x0000ff000000ffffULL, 0x0000ff000000ff00ULL, 0x0000ff000000ff01ULL, 0x0000ff00000000ffULL, 0x0000ff0000000000ULL, 0x0000ff0000000001ULL,
    0x0000ff00000001ffULL, 0x0000ff0000000100ULL, 0x0000ff0000000101ULL, 0x0000ff000001ff00ULL, 0x0000ff00000100ffULL, 0x0000ff0000010000ULL, 0x0000ff0000010001ULL, 0x0000ff0000010100ULL,
    0x0000ff0001ffff01ULL, 0x0000ff0001ff0000ULL, 0x0000ff000100ff00ULL, 0x0000ff00010000ffULL, 0x0000ff0001000000ULL, 0x0000ff0001000001ULL, 0x0000ff0001000100ULL, 0x0000ff000101ffffULL,
    0x0000ff0001010000ULL, 0x0000ff0001010101ULL, 0x0000ff01ffffff00ULL, 0x0000ff01ffff0001ULL, 0x0000ff01ff00ff01ULL, 0x0000ff01ff000000ULL, 0x0000ff01ff000101ULL, 0x0000ff01ff01ff00ULL,
    0x0000ff01ff0100ffULL, 0x0000ff0100ffff01ULL, 0x0000ff0100ff0000ULL, 0x0000ff0100ff0101ULL, 0x0000ff010000ff00ULL, 0x0000ff01000000ffULL, 0x0000ff0100000000ULL, 0x0000ff0100000001ULL,
    0x0000ff0100000100ULL, 0x0000ff010001ff01ULL, 0x0000ff0100010000ULL, 0x0000ff0101ff0000ULL, 0x0000ff010100ffffULL, 0x0000ff010100ff01ULL, 0x0000ff0101000000ULL, 0x0000ff0101000100ULL,
    0x0000ff0101000101ULL, 0x0000ff01010100ffULL, 0x000000ffffff00ffULL, 0x000000ffffff0000ULL, 0x000000ffff00ff00ULL, 0x000000ffff0000ffULL, 0x000000ffff000000ULL, 0x000000ffff000001ULL,
    0x000000ffff0001ffULL, 0x000000ffff000100ULL, 0x000000ffff01ff00ULL, 0x000000ffff010000ULL, 0x000000ffff0101ffULL, 0x000000ffff010101ULL, 0x000000ff00ffff00ULL, 0x000000ff00ff00ffULL,
    0x000000ff00ff0000ULL, 0x000000ff00ff0001ULL, 0x000000ff00ff0100ULL, 0x000000ff00ff0101ULL, 0x000000ff0000ffffULL, 0x000000ff0000ff00ULL, 0x000000ff000000ffULL, 0x000000ff00000000ULL,
    0x000000ff00000001ULL, 0x000000ff000001ffULL, 0x000000ff00000100ULL, 0x000000ff00000101ULL, 0x000000ff0001ff00ULL, 0x000000ff0001ff01ULL, 0x000000ff000100ffULL, 0x000000ff00010000ULL,
    0x000000ff00010001ULL, 0x000000ff00010100ULL, 0x000000ff01ffffffULL, 0x000000ff01ff01ffULL, 0x000000ff01ff0101ULL, 0x000000ff0100ff00ULL, 0x000000ff010000ffULL, 0x000000ff01000000ULL,
    0x000000ff01000001ULL, 0x000000ff01000100ULL, 0x000000ff0101ff00ULL, 0x000000ff010100ffULL, 0x000000ff01010000ULL, 0x000000ff01010101ULL, 0x00000000ffffff00ULL, 0x00000000ffffff01ULL,
    0x00000000ffff00ffULL, 0x00000000ffff0000ULL, 0x00000000ffff0001ULL, 0x00000000ffff0100ULL, 0x00000000ff00ffffULL, 0x00000000ff00ff00ULL, 0x00000000ff00ff01ULL, 0x00000000ff0000ffULL,
    0x00000000ff000000ULL, 0x00000000ff000001ULL, 0x00000000ff000100ULL, 0x00000000ff000101ULL, 0x00000000ff01ff00ULL, 0x00000000ff0100ffULL, 0x00000000ff010000ULL, 0x00000000ff010001ULL,
    0x00000000ff010100ULL, 0x0000000000ffffffULL, 0x0000000000ffff00ULL, 0x0000000000ffff01ULL, 0x0000000000ff00ffULL, 0x0000000000ff0000ULL, 0x0000000000ff0001ULL, 0x0000000000ff01ffULL,
    0x0000000000ff0100ULL, 0x000000000000ffffULL, 0x000000000000ff00ULL, 0x000000000000ff01ULL, 0x00000000000000ffULL, 0x0000000000000000ULL, 0x0000000000000001ULL, 0x00000000000001ffULL,
    0x0000000000000100ULL, 0x0000000000000101ULL, 0x000000000001ffffULL, 0x000000000001ff00ULL, 0x00000000000100ffULL, 0x0000000000010000ULL, 0x0000000000010001ULL, 0x00000000000101ffULL,
    0x0000000000010100ULL, 0x0000000000010101ULL, 0x0000000001ffff00ULL, 0x0000000001ff00ffULL, 0x0000000001ff0000ULL, 0x0000000001ff0100ULL, 0x0000000001ff0101ULL, 0x000000000100ffffULL,
    0x000000000100ff00ULL, 0x00000000010000ffULL, 0x0000000001000000ULL, 0x0000000001000001ULL, 0x00000000010001ffULL, 0x0000000001000100ULL, 0x000000000101ff00ULL, 0x00000000010100ffULL,
    0x0000000001010000ULL, 0x0000000001010001ULL, 0x0000000001010100ULL, 0x00000001ffffffffULL, 0x00000001ffffff00ULL, 0x00000001ffffff01ULL, 0x00000001ffff00ffULL, 0x00000001ffff0001ULL,
    0x00000001ffff01ffULL, 0x00000001ffff0100ULL, 0x00000001ff00ff00ULL, 0x00000001ff0000ffULL, 0x00000001ff000000ULL, 0x00000001ff0001ffULL, 0x00000001ff000100ULL, 0x00000001ff01ffffULL,
    0x00000001ff01ff00ULL, 0x00000001ff01ff01ULL, 0x00000001ff0100ffULL, 0x00000001ff010000ULL, 0x00000001ff010001ULL, 0x00000001ff0101ffULL, 0x00000001ff010100ULL, 0x0000000100ffff00ULL,
    0x0000000100ff0000ULL, 0x0000000100ff0001ULL, 0x0000000100ff01ffULL, 0x0000000100ff0100ULL, 0x0000000100ff0101ULL, 0x000000010000ffffULL, 0x000000010000ff00ULL, 0x000000010000ff01ULL,
    0x00000001000000ffULL, 0x0000000100000000ULL, 0x0000000100000001ULL, 0x00000001000001ffULL, 0x0000000100000100ULL, 0x0000000100000101ULL, 0x000000010001ff00ULL, 0x00000001000100ffULL,
    0x0000000100010000ULL, 0x0000000100010100ULL, 0x0000000101ffff01ULL, 0x0000000101ff0000ULL, 0x0000000101ff0001ULL, 0x0000000101ff01ffULL, 0x0000000101ff0100ULL, 0x0000000101ff0101ULL,
    0x000000010100ff00ULL, 0x0000000101000000ULL, 0x0000000101000101ULL, 0x000000010101ff01ULL, 0x0000000101010000ULL, 0x0000000101010001ULL, 0x00000001010101ffULL, 0x0000000101010100ULL,
    0x000001ffffff00ffULL, 0x000001ffffff0000ULL, 0x000001ffffff0001ULL, 0x000001ffffff0100ULL, 0x000001ffff00ffffULL, 0x000001ffff000000ULL, 0x000001ffff0001ffULL, 0x000001ffff01ff00ULL,
    0x000001ffff010101ULL, 0x000001ff00ff0000ULL, 0x000001ff00ff01ffULL, 0x000001ff00ff0101ULL, 0x000001ff0000ff00ULL, 0x000001ff000000ffULL, 0x000001ff00000000ULL, 0x000001ff00000001ULL,
    0x000001ff000001ffULL, 0x000001ff00000100ULL, 0x000001ff0001ffffULL, 0x000001ff0001ff01ULL, 0x000001ff000100ffULL, 0x000001ff00010000ULL, 0x000001ff01ffff01ULL, 0x000001ff01ff0100ULL,
    0x000001ff0100ffffULL, 0x000001ff0100ff01ULL, 0x000001ff01000000ULL, 0x000001ff010001ffULL, 0x000001ff0101ff00ULL, 0x000001ff01010100ULL, 0x00000100ffffff00ULL, 0x00000100ffffff01ULL,
    0x00000100ffff0000ULL, 0x00000100ffff0101ULL, 0x00000100ff00ff00ULL, 0x00000100ff0000ffULL, 0x00000100ff000000ULL, 0x00000100ff000001ULL, 0x00000100ff000100ULL, 0x00000100ff010000ULL,
    0x0000010000ffff00ULL, 0x0000010000ff00ffULL, 0x0000010000ff0000ULL, 0x0000010000ff0001ULL, 0x0000010000ff0100ULL, 0x000001000000ffffULL, 0x000001000000ff00ULL, 0x000001000000ff01ULL,
    0x00000100000000ffULL, 0x0000010000000000ULL, 0x0000010000000001ULL, 0x00000100000001ffULL, 0x0000010000000100ULL, 0x0000010000000101ULL, 0x000001000001ff00ULL, 0x00000100000100ffULL,
    0x0000010000010000ULL, 0x0000010000010001ULL, 0x0000010000010100ULL, 0x0000010001ffff00ULL, 0x0000010001ff0000ULL, 0x0000010001ff0100ULL, 0x000001000100ff00ULL, 0x00000100010000ffULL,
    0x0000010001000000ULL, 0x0000010001000001ULL, 0x00000100010001ffULL, 0x0000010001000100ULL, 0x0000010001010000ULL, 0x00000101ffff00ffULL, 0x00000101ffff01ffULL, 0x00000101ff000000ULL,
    0x00000101ff000101ULL, 0x00000101ff01ffffULL, 0x00000101ff010000ULL, 0x00000101ff010001ULL, 0x00000101ff010100ULL, 0x0000010100ff0000ULL, 0x0000010100ff01ffULL, 0x0000010100ff0100ULL,
    0x000001010000ff00ULL, 0x0000010100000000ULL, 0x0000010100000001ULL, 0x00000101000001ffULL, 0x0000010100000100ULL, 0x000001010001ff01ULL, 0x0000010100010000ULL, 0x00000101000101ffULL,
    0x0000010100010101ULL, 0x0000010101ffff00ULL, 0x0000010101ff0101ULL, 0x000001010100ff01ULL, 0x0000010101000000ULL, 0x0000010101000001ULL, 0x00000101010001ffULL, 0x0000010101000101ULL,
    0x000001010101ff00ULL, 0x0001ffffffff0000ULL, 0x0001ffffff0000ffULL, 0x0001ffffff000001ULL, 0x0001ffffff000100ULL, 0x0001ffffff010000ULL, 0x0001ffff00ff00ffULL, 0x0001ffff0000ffffULL,
    0x0001ffff00000000ULL, 0x0001ffff00000001ULL, 0x0001ffff000001ffULL, 0x0001ffff00000101ULL, 0x0001ffff0001ff00ULL, 0x0001ffff000100ffULL, 0x0001ffff00010001ULL, 0x0001ffff00010100ULL,
    0x0001ffff01ffff00ULL, 0x0001ffff01000001ULL, 0x0001ffff01010000ULL, 0x0001ff00ffffff00ULL, 0x0001ff00ffff00ffULL, 0x0001ff00ffff0001ULL, 0x0001ff00ffff0100ULL, 0x0001ff00ff00ff01ULL,
    0x0001ff00ff000000ULL, 0x0001ff00ff01ff00ULL, 0x0001ff00ff01ff01ULL, 0x0001ff00ff010001ULL, 0x0001ff00ff010100ULL, 0x0001ff0000ff0000ULL, 0x0001ff0000ff0100ULL, 0x0001ff000000ff00ULL,
    0x0001ff0000000000ULL, 0x0001ff0000000001ULL, 0x0001ff0000000100ULL, 0x0001ff0000010000ULL, 0x0001ff0000010001ULL, 0x0001ff0000010101ULL, 0x0001ff0001ff00ffULL, 0x0001ff0001ff0101ULL,
    0x0001ff000100ff01ULL, 0x0001ff0001000000ULL, 0x0001ff000101ff00ULL, 0x0001ff0001010001ULL, 0x0001ff0001010100ULL, 0x0001ff01ff00ff00ULL, 0x0001ff01ff000001ULL, 0x0001ff01ff000100ULL,
    0x0001ff0100ffffffULL, 0x0001ff0100ffff00ULL, 0x0001ff0100ff0001ULL, 0x0001ff0100000000ULL, 0x0001ff0100000001ULL, 0x0001ff01000001ffULL, 0x0001ff010001ffffULL, 0x0001ff0101ff0000ULL,
    0x0001ff010100ff00ULL, 0x0001ff0101000001ULL, 0x0001ff0101010000ULL, 0x000100ffff00ff00ULL, 0x000100ffff00ff01ULL, 0x000100ffff000000ULL, 0x000100ffff000001ULL, 0x000100ffff000101ULL,
    0x000100ffff01ff00ULL, 0x000100ffff010001ULL, 0x000100ffff010100ULL, 0x000100ff00ffffffULL, 0x000100ff00ffff01ULL, 0x000100ff00ff0000ULL, 0x000100ff00ff01ffULL, 0x000100ff00ff0101ULL,
    0x000100ff0000ff00ULL, 0x000100ff000000ffULL, 0x000100ff00000000ULL, 0x000100ff00000001ULL, 0x000100ff00000100ULL, 0x000100ff00000101ULL, 0x000100ff0001ffffULL, 0x000100ff0001ff01ULL,
    0x000100ff00010000ULL, 0x000100ff01ff00ffULL, 0x000100ff01ff0000ULL, 0x000100ff01ff0100ULL, 0x000100ff0100ffffULL, 0x000100ff0100ff01ULL, 0x000100ff010000ffULL, 0x000100ff01000000ULL,
    0x000100ff01000001ULL, 0x000100ff010001ffULL, 0x000100ff01000101ULL, 0x000100ff0101ff00ULL, 0x000100ff010100ffULL, 0x000100ff01010100ULL, 0x00010000ffff0000ULL, 0x00010000ffff01ffULL,
    0x00010000ffff0101ULL, 0x00010000ff00ff00ULL, 0x00010000ff000000ULL, 0x00010000ff000001ULL, 0x00010000ff000100ULL, 0x0001000000ff00ffULL, 0x0001000000ff0000ULL, 0x0001000000ff0001ULL,
    0x0001000000ff0100ULL, 0x000100000000ffffULL, 0x000100000000ff00ULL, 0x00010000000000ffULL, 0x0001000000000000ULL, 0x0001000000000001ULL, 0x0001000000000100ULL, 0x000100000001ff00ULL,
    0x00010000000100ffULL, 0x0001000000010000ULL, 0x0001000000010001ULL, 0x0001000000010100ULL, 0x0001000001ff0001ULL, 0x0001000001ff0100ULL, 0x0001000001ff0101ULL, 0x000100000100ff00ULL,
    0x0001000001000000ULL, 0x0001000001000001ULL, 0x0001000001000100ULL, 0x0001000001000101ULL, 0x000100000101ff01ULL, 0x0001000001010000ULL, 0x0001000001010001ULL, 0x00010000010101ffULL,
    0x00010001ffffff01ULL, 0x00010001ffff0100ULL, 0x00010001ff000000ULL, 0x00010001ff01ffffULL, 0x00010001ff010001ULL, 0x00010001ff0101ffULL, 0x00010001ff010100ULL, 0x0001000100ffffffULL,
    0x0001000100ff0000ULL, 0x0001000100ff01ffULL, 0x0001000100ff0101ULL, 0x000100010000ff00ULL, 0x00010001000000ffULL, 0x0001000100000000ULL, 0x0001000100000001ULL, 0x00010001000001ffULL,
    0x0001000100000101ULL, 0x000100010001ffffULL, 0x0001000100010000ULL, 0x00010001000101ffULL, 0x0001000101ffffffULL, 0x0001000101ffff01ULL, 0x0001000101ff0000ULL, 0x0001000101ff0101ULL,
    0x00010001010000ffULL, 0x0001000101000001ULL, 0x00010001010001ffULL, 0x0001000101000100ULL, 0x000100010101ffffULL, 0x00010001010100ffULL, 0x0001000101010001ULL, 0x0001000101010101ULL,
    0x000101ffff000001ULL, 0x000101ffff000100ULL, 0x000101ffff010000ULL, 0x000101ff00ffff00ULL, 0x000101ff0000ff01ULL, 0x000101ff00000000ULL, 0x000101ff00000101ULL, 0x000101ff0001ff00ULL,
    0x000101ff00010100ULL, 0x000101ff01ff0000ULL, 0x000101ff0100ff00ULL, 0x000101ff010001ffULL, 0x000101ff01010001ULL, 0x00010100ffffff00ULL, 0x00010100ffff00ffULL, 0x00010100ff00ffffULL,
    0x00010100ff000000ULL, 0x00010100ff01ff00ULL, 0x00010100ff0100ffULL, 0x00010100ff010001ULL, 0x00010100ff010100ULL, 0x0001010000ffffffULL, 0x0001010000ffff00ULL, 0x0001010000ff0000ULL,
    0x0001010000ff0001ULL, 0x0001010000ff01ffULL, 0x000101000000ff00ULL, 0x00010100000000ffULL, 0x0001010000000000ULL, 0x0001010000000001ULL, 0x0001010000000100ULL, 0x000101000001ffffULL,
    0x0001010000010000ULL, 0x0001010000010101ULL, 0x0001010001ffff01ULL, 0x0001010001ff00ffULL, 0x0001010001ff0101ULL, 0x0001010001000000ULL, 0x000101000101ff00ULL, 0x00010100010100ffULL,
    0x0001010001010000ULL, 0x0001010001010100ULL, 0x00010101ff00ff00ULL, 0x00010101ff000001ULL, 0x00010101ff0001ffULL, 0x0001010100ffff00ULL, 0x0001010100ff00ffULL, 0x0001010100ff0100ULL,
    0x000101010000ffffULL, 0x0001010100000000ULL, 0x00010101000001ffULL, 0x0001010100000101ULL, 0x00010101000100ffULL, 0x0001010100010000ULL, 0x0001010100010100ULL, 0x0001010101ff0001ULL,
    0x00010101010000ffULL, 0x00010101010001ffULL, 0x0001010101000101ULL, 0x0001010101010001ULL, 0x01ffffffffffffffULL, 0x01ffffffffffff01ULL, 0x01ffffffffff01ffULL, 0x01ffffffffff0101ULL,
    0x01ffffffff01ffffULL, 0x01ffffffff01ff01ULL, 0x01ffffffff0101ffULL, 0x01ffffffff010101ULL, 0x01ffffff00ff0000ULL, 0x01ffffff0000ffffULL, 0x01ffffff0000ff00ULL, 0x01ffffff000000ffULL,
    0x01ffffff00000001ULL, 0x01ffffff00000100ULL, 0x01ffffff00010000ULL, 0x01ffffff01ffffffULL, 0x01ffffff01ffff01ULL, 0x01ffffff01ff01ffULL, 0x01ffffff01ff0101ULL, 0x01ffffff01000000ULL,
    0x01ffffff0101ffffULL, 0x01ffffff0101ff01ULL, 0x01ffffff010101ffULL, 0x01ffffff01010101ULL, 0x01ffff00ffff0000ULL, 0x01ffff00ff00ff00ULL, 0x01ffff00ff0000ffULL, 0x01ffff00ff000001ULL,
    0x01ffff00ff000100ULL, 0x01ffff00ff010000ULL, 0x01ffff0000ffff00ULL, 0x01ffff0000ff00ffULL, 0x01ffff0000ff0100ULL, 0x01ffff000000ffffULL, 0x01ffff000000ff01ULL, 0x01ffff0000000000ULL,
    0x01ffff0000000001ULL, 0x01ffff00000001ffULL, 0x01ffff0000000100ULL, 0x01ffff00000100ffULL, 0x01ffff0000010001ULL, 0x01ffff0000010100ULL, 0x01ffff0001ff0000ULL, 0x01ffff0001ff0100ULL,
    0x01ffff00010000ffULL, 0x01ffff0001000001ULL, 0x01ffff0001000100ULL, 0x01ffff0001010000ULL, 0x01ffff01ffffffffULL, 0x01ffff01ffffff01ULL, 0x01ffff01ffff01ffULL, 0x01ffff01ffff0101ULL,
    0x01ffff01ff000000ULL, 0x01ffff01ff01ffffULL, 0x01ffff01ff01ff01ULL, 0x01ffff01ff0101ffULL, 0x01ffff01ff010101ULL, 0x01ffff010000ff00ULL, 0x01ffff01000000ffULL, 0x01ffff0100000100ULL,
    0x01ffff0100010000ULL, 0x01ffff0101ffffffULL, 0x01ffff0101ffff01ULL, 0x01ffff0101ff01ffULL, 0x01ffff0101ff0101ULL, 0x01ffff0101000000ULL, 0x01ffff010101ffffULL, 0x01ffff010101ff01ULL,
    0x01ffff01010101ffULL, 0x01ffff0101010101ULL, 0x01ff00ffff0000ffULL, 0x01ff00ffff000100ULL, 0x01ff00ff00ffff00ULL, 0x01ff00ff00ff00ffULL, 0x01ff00ff0000ff00ULL, 0x01ff00ff00000000ULL,
    0x01ff00ff00000101ULL, 0x01ff00ff0001ff00ULL, 0x01ff00ff000100ffULL, 0x01ff00ff00010100ULL, 0x01ff00ff010000ffULL, 0x01ff00ff01000100ULL, 0x01ff0000ffffff00ULL, 0x01ff0000ffff0100ULL,
    0x01ff0000ff00ff01ULL, 0x01ff0000ff000000ULL, 0x01ff0000ff000101ULL, 0x01ff0000ff010001ULL, 0x01ff0000ff010100ULL, 0x01ff000000ffffffULL, 0x01ff000000ffff00ULL, 0x01ff000000ff0000ULL,
    0x01ff000000ff01ffULL, 0x01ff00000000ff00ULL, 0x01ff0000000000ffULL, 0x01ff000000000000ULL, 0x01ff000000000001ULL, 0x01ff000000000100ULL, 0x01ff000000000101ULL, 0x01ff000000010000ULL,
    0x01ff000000010001ULL, 0x01ff0000000101ffULL, 0x01ff000000010101ULL, 0x01ff000001ffff00ULL, 0x01ff000001ff00ffULL, 0x01ff000001ff0001ULL, 0x01ff000001ff0100ULL, 0x01ff00000100ffffULL,
    0x01ff00000100ff01ULL, 0x01ff000001000000ULL, 0x01ff0000010001ffULL, 0x01ff000001010001ULL, 0x01ff0001ff00ff00ULL, 0x01ff0001ff000001ULL, 0x01ff0001ff000100ULL, 0x01ff0001ff010000ULL,
    0x01ff000100ffff00ULL, 0x01ff000100ff00ffULL, 0x01ff000100ff0100ULL, 0x01ff000100ff0101ULL, 0x01ff00010000ffffULL, 0x01ff000100000000ULL, 0x01ff000100000100ULL, 0x01ff000100000101ULL,
    0x01ff00010001ff00ULL, 0x01ff000100010001ULL, 0x01ff000100010101ULL, 0x01ff000101ff0000ULL, 0x01ff00010100ff00ULL, 0x01ff000101000101ULL, 0x01ff0001010100ffULL, 0x01ff01ffffffffffULL,
    0x01ff01ffffffff01ULL, 0x01ff01ffffff01ffULL, 0x01ff01ffffff0101ULL, 0x01ff01ffff000000ULL, 0x01ff01ffff01ffffULL, 0x01ff01ffff01ff01ULL, 0x01ff01ffff0101ffULL, 0x01ff01ffff010101ULL,
    0x01ff01ff00ffff00ULL, 0x01ff01ff00ff0000ULL, 0x01ff01ff0000ff00ULL, 0x01ff01ff000000ffULL, 0x01ff01ff00000100ULL, 0x01ff01ff00010000ULL, 0x01ff01ff00010100ULL, 0x01ff01ff01ffffffULL,
    0x01ff01ff01ffff01ULL, 0x01ff01ff01ff01ffULL, 0x01ff01ff01ff0101ULL, 0x01ff01ff01000000ULL, 0x01ff01ff0101ffffULL, 0x01ff01ff0101ff01ULL, 0x01ff01ff010101ffULL, 0x01ff01ff01010101ULL,
    0x01ff0100ffff0000ULL, 0x01ff0100ffff0001ULL, 0x01ff0100ff00ff00ULL, 0x01ff0100ff0000ffULL, 0x01ff0100ff000001ULL, 0x01ff0100ff010000ULL, 0x01ff010000ffff00ULL, 0x01ff010000ff00ffULL,
    0x01ff010000ff0001ULL, 0x01ff010000ff0100ULL, 0x01ff01000000ffffULL, 0x01ff01000000ff01ULL, 0x01ff010000000000ULL, 0x01ff010000000101ULL, 0x01ff01000001ff00ULL, 0x01ff0100000100ffULL,
    0x01ff010001ff0000ULL, 0x01ff010001000001ULL, 0x01ff010001000100ULL, 0x01ff010001010000ULL, 0x01ff0101ffffffffULL, 0x01ff0101ffffff01ULL, 0x01ff0101ffff01ffULL, 0x01ff0101ffff0101ULL,
    0x01ff0101ff000000ULL, 0x01ff0101ff01ffffULL, 0x01ff0101ff01ff01ULL, 0x01ff0101ff0101ffULL, 0x01ff0101ff010101ULL, 0x01ff010100ff0000ULL, 0x01ff01010000ff00ULL, 0x01ff0101000000ffULL,
    0x01ff010100000001ULL, 0x01ff010101ffffffULL, 0x01ff010101ffff01ULL, 0x01ff010101ff01ffULL, 0x01ff010101ff0101ULL, 0x01ff010101000000ULL, 0x01ff01010101ffffULL, 0x01ff01010101ff01ULL,
    0x01ff0101010101ffULL, 0x01ff010101010101ULL, 0x0100ffffffff0000ULL, 0x0100ffffff00ff00ULL, 0x0100ffffff000001ULL, 0x0100ffffff0001ffULL, 0x0100ffffff000100ULL, 0x0100ffffff010000ULL,
    0x0100ffff00ffff00ULL, 0x0100ffff00ff0001ULL, 0x0100ffff00ff0100ULL, 0x0100ffff00000000ULL, 0x0100ffff000001ffULL, 0x0100ffff00000101ULL, 0x0100ffff00010100ULL, 0x0100ffff00010101ULL,
    0x0100ffff01ff0000ULL, 0x0100ffff0100ff00ULL, 0x0100ffff010000ffULL, 0x0100ffff01000001ULL, 0x0100ffff01000100ULL, 0x0100ffff01010000ULL, 0x0100ff00ffffff00ULL, 0x0100ff00ffff00ffULL,
    0x0100ff00ffff0001ULL, 0x0100ff00ffff0100ULL, 0x0100ff00ff00ffffULL, 0x0100ff00ff000000ULL, 0x0100ff00ff0001ffULL, 0x0100ff00ff000101ULL, 0x0100ff00ff01ff00ULL, 0x0100ff00ff0100ffULL,
    0x0100ff00ff010001ULL, 0x0100ff00ff010100ULL, 0x0100ff0000ffffffULL, 0x0100ff0000ff0000ULL, 0x0100ff000000ffffULL, 0x0100ff000000ff00ULL, 0x0100ff00000000ffULL, 0x0100ff0000000000ULL,
    0x0100ff0000000001ULL, 0x0100ff0000000100ULL, 0x0100ff000001ff01ULL, 0x0100ff0000010000ULL, 0x0100ff0001ff00ffULL, 0x0100ff0001ff0001ULL, 0x0100ff000100ff01ULL, 0x0100ff0001000000ULL,
    0x0100ff00010001ffULL, 0x0100ff000101ff00ULL, 0x0100ff00010100ffULL, 0x0100ff0001010001ULL, 0x0100ff0001010100ULL, 0x0100ff01ffff0000ULL, 0x0100ff01ff00ff00ULL, 0x0100ff01ff0000ffULL,
    0x0100ff01ff000100ULL, 0x0100ff01ff010000ULL, 0x0100ff0100ff00ffULL, 0x0100ff0100ff0001ULL, 0x0100ff0100ff0100ULL, 0x0100ff010000ffffULL, 0x0100ff010000ff01ULL, 0x0100ff0100000000ULL,
    0x0100ff01000001ffULL, 0x0100ff0100010001ULL, 0x0100ff0100010100ULL, 0x0100ff0101ff0000ULL, 0x0100ff01010000ffULL, 0x0100ff0101000001ULL, 0x0100ff0101010100ULL, 0x010000ffffffff00ULL,
    0x010000ffffff00ffULL, 0x010000ffffff0001ULL, 0x010000ffff00ffffULL, 0x010000ffff000000ULL, 0x010000ffff0001ffULL, 0x010000ffff010001ULL, 0x010000ff00ffffffULL, 0x010000ff00ff0101ULL,
    0x010000ff0000ff00ULL, 0x010000ff000000ffULL, 0x010000ff00000000ULL, 0x010000ff00000001ULL, 0x010000ff000001ffULL, 0x010000ff00000100ULL, 0x010000ff0001ffffULL, 0x010000ff0001ff00ULL,
    0x010000ff0001ff01ULL, 0x010000ff00010000ULL, 0x010000ff01ff00ffULL, 0x010000ff01ff0001ULL, 0x010000ff0100ff01ULL, 0x010000ff010000ffULL, 0x010000ff01000000ULL, 0x010000ff010001ffULL,
    0x010000ff0101ff00ULL, 0x010000ff01010100ULL, 0x01000000ffffffffULL, 0x01000000ffff0000ULL, 0x01000000ffff01ffULL, 0x01000000ffff0101ULL, 0x01000000ff00ffffULL, 0x01000000ff00ff00ULL,
    0x01000000ff0000ffULL, 0x01000000ff000000ULL, 0x01000000ff000001ULL, 0x01000000ff000100ULL, 0x01000000ff01ff00ULL, 0x01000000ff010000ULL, 0x01000000ff010100ULL, 0x01000000ff010101ULL,
    0x0100000000ffff00ULL, 0x0100000000ff00ffULL, 0x0100000000ff0000ULL, 0x0100000000ff0001ULL, 0x0100000000ff0100ULL, 0x010000000000ffffULL, 0x010000000000ff00ULL, 0x010000000000ff01ULL,
    0x01000000000000ffULL, 0x0100000000000000ULL, 0x0100000000000001ULL, 0x01000000000001ffULL, 0x0100000000000100ULL, 0x0100000000000101ULL, 0x010000000001ff00ULL, 0x01000000000100ffULL,
    0x0100000000010000ULL, 0x0100000000010001ULL, 0x0100000000010100ULL, 0x0100000001ffff00ULL, 0x0100000001ff0000ULL, 0x0100000001ff01ffULL, 0x010000000100ff00ULL, 0x010000000100ff01ULL,
    0x01000000010000ffULL, 0x0100000001000000ULL, 0x0100000001000001ULL, 0x0100000001000100ULL, 0x0100000001000101ULL, 0x010000000101ffffULL, 0x010000000101ff01ULL, 0x0100000001010000ULL,
    0x01000000010101ffULL, 0x0100000001010101ULL, 0x01000001ffffff00ULL, 0x01000001ffff00ffULL, 0x01000001ff00ffffULL, 0x01000001ff000000ULL, 0x01000001ff000100ULL, 0x01000001ff01ffffULL,
    0x01000001ff010001ULL, 0x01000001ff010100ULL, 0x0100000100ff0000ULL, 0x0100000100ff01ffULL, 0x0100000100ff0100ULL, 0x010000010000ff00ULL, 0x010000010000ff01ULL, 0x0100000100000000ULL,
    0x0100000100000001ULL, 0x0100000100000100ULL, 0x0100000100010000ULL, 0x01000001000101ffULL, 0x0100000101ffff01ULL, 0x0100000101ff00ffULL, 0x0100000101ff0100ULL, 0x0100000101ff0101ULL,
    0x010000010100ff01ULL, 0x01000001010000ffULL, 0x0100000101000000ULL, 0x01000001010100ffULL, 0x0100000101010001ULL, 0x0100000101010100ULL, 0x010001ffffff0000ULL, 0x010001ffff000001ULL,
    0x010001ffff000100ULL, 0x010001ffff010000ULL, 0x010001ff00ffff00ULL, 0x010001ff00ff0001ULL, 0x010001ff0000ffffULL, 0x010001ff0000ff01ULL, 0x010001ff00000000ULL, 0x010001ff00000001ULL,
    0x010001ff00000101ULL, 0x010001ff000100ffULL, 0x010001ff00010000ULL, 0x010001ff01ff0000ULL, 0x010001ff0100ff00ULL, 0x010001ff01000001ULL, 0x010001ff01000100ULL, 0x010001ff01010000ULL,
    0x01000100ffff00ffULL, 0x01000100ffff0001ULL, 0x01000100ffff0100ULL, 0x01000100ff00ffffULL, 0x01000100ff00ff01ULL, 0x01000100ff000000ULL, 0x01000100ff0001ffULL, 0x01000100ff000101ULL,
    0x01000100ff01ffffULL, 0x01000100ff01ff00ULL, 0x01000100ff0100ffULL, 0x01000100ff010001ULL, 0x0100010000ffffffULL, 0x0100010000ffff01ULL, 0x0100010000ff0000ULL, 0x0100010000ff01ffULL,
    0x0100010000ff0101ULL, 0x010001000000ff00ULL, 0x01000100000000ffULL, 0x0100010000000000ULL, 0x0100010000000001ULL, 0x0100010000000100ULL, 0x010001000001ff01ULL, 0x0100010000010000ULL,
    0x0100010000010001ULL, 0x0100010000010101ULL, 0x0100010001ffff00ULL, 0x0100010001ff00ffULL, 0x010001000100ffffULL, 0x010001000100ff01ULL, 0x0100010001000000ULL, 0x0100010001000101ULL,
    0x010001000101ff00ULL, 0x0100010001010001ULL, 0x01000101ffff0000ULL, 0x01000101ff000000ULL, 0x01000101ff010000ULL, 0x0100010100ff00ffULL, 0x0100010100ff0001ULL, 0x0100010100ff0100ULL,
    0x010001010000ffffULL, 0x0100010100000000ULL, 0x01000101000001ffULL, 0x010001010001ff00ULL, 0x0100010101ff0000ULL, 0x010001010100ff00ULL, 0x01000101010000ffULL, 0x0100010101000000ULL,
    0x0100010101000001ULL, 0x0101ffffffffffffULL, 0x0101ffffffffff01ULL, 0x0101ffffffff01ffULL, 0x0101ffffffff0101ULL, 0x0101ffffff000000ULL, 0x0101ffffff01ffffULL, 0x0101ffffff01ff01ULL,
    0x0101ffffff0101ffULL, 0x0101ffffff010101ULL, 0x0101ffff00ff0000ULL, 0x0101ffff0000ff00ULL, 0x0101ffff000000ffULL, 0x0101ffff00000001ULL, 0x0101ffff00000100ULL, 0x0101ffff01ffffffULL,
    0x0101ffff01ffff01ULL, 0x0101ffff01ff01ffULL, 0x0101ffff01ff0101ULL, 0x0101ffff01000000ULL, 0x0101ffff0101ffffULL, 0x0101ffff0101ff01ULL, 0x0101ffff010101ffULL, 0x0101ffff01010101ULL,
    0x0101ff00ffff0000ULL, 0x0101ff00ffff0100ULL, 0x0101ff00ff00ff00ULL, 0x0101ff00ff0000ffULL, 0x0101ff00ff000001ULL, 0x0101ff00ff000100ULL, 0x0101ff00ff000101ULL, 0x0101ff0000ff0001ULL,
    0x0101ff0000ff0100ULL, 0x0101ff000000ff00ULL, 0x0101ff0000000000ULL, 0x0101ff00000001ffULL, 0x0101ff0000000101ULL, 0x0101ff000001ff00ULL, 0x0101ff00000100ffULL, 0x0101ff0001ff0000ULL,
    0x0101ff000100ffffULL, 0x0101ff000100ff01ULL, 0x0101ff0001000001ULL, 0x0101ff0001000100ULL, 0x0101ff01ffffff01ULL, 0x0101ff01ffff01ffULL, 0x0101ff01ffff0101ULL, 0x0101ff01ff00ffffULL,
    0x0101ff01ff000100ULL, 0x0101ff01ff01ff01ULL, 0x0101ff01ff0101ffULL, 0x0101ff01ff010101ULL, 0x0101ff0100ff0000ULL, 0x0101ff010000ff00ULL, 0x0101ff0100000001ULL, 0x0101ff0100000100ULL,
    0x0101ff0100010000ULL, 0x0101ff0101ffffffULL, 0x0101ff0101ffff01ULL, 0x0101ff0101ff01ffULL, 0x0101ff0101ff0101ULL, 0x0101ff0101000000ULL, 0x0101ff010101ffffULL, 0x0101ff010101ff01ULL,
    0x0101ff01010101ffULL, 0x0101ff0101010101ULL, 0x010100ffff000100ULL, 0x010100ffff010000ULL, 0x010100ff00ffff00ULL, 0x010100ff00ff00ffULL, 0x010100ff0000ffffULL, 0x010100ff000000ffULL,
    0x010100ff00000000ULL, 0x010100ff000001ffULL, 0x010100ff00000101ULL, 0x010100ff0001ff00ULL, 0x010100ff00010000ULL, 0x010100ff00010001ULL, 0x010100ff000101ffULL, 0x010100ff00010100ULL,
    0x010100ff01ff0000ULL, 0x01010000ffff0001ULL, 0x01010000ffff0100ULL, 0x01010000ff00ffffULL, 0x01010000ff00ff01ULL, 0x01010000ff000000ULL, 0x01010000ff0001ffULL, 0x01010000ff010001ULL,
    0x01010000ff010100ULL, 0x0101000000ffff01ULL, 0x0101000000ff0000ULL, 0x010100000000ff00ULL, 0x01010000000000ffULL, 0x0101000000000000ULL, 0x0101000000000001ULL, 0x0101000000000100ULL,
    0x0101000000010000ULL, 0x0101000000010101ULL, 0x0101000001ffff00ULL, 0x0101000001ff00ffULL, 0x0101000001ff0000ULL, 0x0101000001ff0001ULL, 0x0101000001ff0100ULL, 0x010100000100ff01ULL,
    0x0101000001000000ULL, 0x01010000010001ffULL, 0x01010001ffff0000ULL, 0x01010001ff00ff00ULL, 0x01010001ff000001ULL, 0x01010001ff000101ULL, 0x01010001ff01ff00ULL, 0x01010001ff010000ULL,
    0x0101000100ff00ffULL, 0x0101000100ff0001ULL, 0x0101000100ff0101ULL, 0x010100010000ff01ULL, 0x0101000100000000ULL, 0x0101000100000001ULL, 0x01010001000001ffULL, 0x010100010001ffffULL,
    0x010100010001ff01ULL, 0x0101000101ff0001ULL, 0x010100010100ffffULL, 0x0101000101000000ULL, 0x0101000101000001ULL, 0x0101000101000100ULL, 0x010100010101ff00ULL, 0x01010001010100ffULL,
    0x0101000101010001ULL, 0x010101ffffffffffULL, 0x010101ffffffff01ULL, 0x010101ffffff01ffULL, 0x010101ffffff0101ULL, 0x010101ffff01ffffULL, 0x010101ffff01ff01ULL, 0x010101ffff0101ffULL,
    0x010101ffff010101ULL, 0x010101ff0000ff00ULL, 0x010101ff000000ffULL, 0x010101ff00000001ULL, 0x010101ff00000100ULL, 0x010101ff01ffffffULL, 0x010101ff01ffff01ULL, 0x010101ff01ff01ffULL,
    0x010101ff01ff0101ULL, 0x010101ff01000000ULL, 0x010101ff0101ffffULL, 0x010101ff0101ff01ULL, 0x010101ff010101ffULL, 0x010101ff01010101ULL, 0x01010100ffff0000ULL, 0x01010100ff0000ffULL,
    0x01010100ff000100ULL, 0x01010100ff01ff00ULL, 0x01010100ff010000ULL, 0x0101010000ffff00ULL, 0x010101000000ffffULL, 0x0101010000000000ULL, 0x0101010000000101ULL, 0x010101000001ff00ULL,
    0x0101010000010001ULL, 0x0101010000010100ULL, 0x010101000100ffffULL, 0x0101010001000001ULL, 0x01010101ffffffffULL, 0x01010101ffffff01ULL, 0x01010101ffff01ffULL, 0x01010101ffff0101ULL,
    0x01010101ff01ffffULL, 0x01010101ff01ff01ULL, 0x01010101ff0101ffULL, 0x01010101ff010101ULL, 0x010101010000ff00ULL, 0x01010101000000ffULL, 0x0101010100000001ULL, 0x0101010101ffffffULL,
    0x0101010101ffff01ULL, 0x0101010101ff01ffULL, 0x0101010101ff0101ULL, 0x0101010101000000ULL, 0x010101010101ffffULL, 0x010101010101ff01ULL, 0x01010101010101ffULL, 0x0101010101010101ULL,
};
#endif

// AUTO-GENERATED from hanzo-ml/src/quantized/iq_grids.rs (the CPU to_float oracle's codebooks).
// DO NOT EDIT BY HAND. Regenerate with scratchpad/gen_grids.py. Each value is a textual copy
// of the matching Rust literal; the generator asserts token count == declared array length.
#ifdef __HIPCC__
__device__ __constant__ uint64_t IQ2XXS_GRID_D[256] = {
    0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL, 0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x08080808082b0808ULL,
    0x08080808082b082bULL, 0x08080808082b2b08ULL, 0x08080808082b2b2bULL, 0x0808080819080819ULL, 0x0808080819081908ULL, 0x0808080819190808ULL, 0x0808080819192b08ULL, 0x08080808192b0819ULL,
    0x08080808192b1908ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b082b2bULL, 0x080808082b2b082bULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x0808081908190808ULL,
    0x0808081908191919ULL, 0x0808081919080808ULL, 0x080808192b081908ULL, 0x080808192b192b08ULL, 0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b082b082bULL, 0x0808082b2b08082bULL,
    0x0808190808080819ULL, 0x0808190808081908ULL, 0x0808190808190808ULL, 0x08081908082b0819ULL, 0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819082b08ULL,
    0x08081908192b0808ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL, 0x080819082b2b1908ULL, 0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908082b08ULL,
    0x08081919082b0808ULL, 0x080819191908192bULL, 0x08081919192b2b19ULL, 0x080819192b080808ULL, 0x080819192b190819ULL, 0x0808192b08082b19ULL, 0x0808192b08190808ULL, 0x0808192b19080808ULL,
    0x0808192b2b081908ULL, 0x0808192b2b2b1908ULL, 0x08082b0808080808ULL, 0x08082b0808081919ULL, 0x08082b0808082b08ULL, 0x08082b0808191908ULL, 0x08082b08082b2b08ULL, 0x08082b0819080819ULL,
    0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b081919082bULL, 0x08082b082b082b08ULL, 0x08082b1908081908ULL, 0x08082b1919080808ULL, 0x08082b2b0808082bULL, 0x08082b2b08191908ULL,
    0x0819080808080819ULL, 0x0819080808081908ULL, 0x0819080808190808ULL, 0x08190808082b0819ULL, 0x0819080819080808ULL, 0x08190808192b0808ULL, 0x081908082b081908ULL, 0x081908082b190808ULL,
    0x081908082b191919ULL, 0x0819081908080808ULL, 0x0819081908082b08ULL, 0x08190819082b0808ULL, 0x0819081919190808ULL, 0x0819081919192b2bULL, 0x081908192b080808ULL, 0x0819082b082b1908ULL,
    0x0819082b19081919ULL, 0x0819190808080808ULL, 0x0819190808082b08ULL, 0x08191908082b0808ULL, 0x08191908082b1919ULL, 0x0819190819082b19ULL, 0x081919082b080808ULL, 0x0819191908192b08ULL,
    0x08191919192b082bULL, 0x0819192b08080808ULL, 0x0819192b0819192bULL, 0x08192b0808080819ULL, 0x08192b0808081908ULL, 0x08192b0808190808ULL, 0x08192b0819080808ULL, 0x08192b082b080819ULL,
    0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b192b2b0808ULL, 0x08192b2b19190819ULL, 0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808082b2bULL, 0x082b080819081908ULL,
    0x082b0808192b0819ULL, 0x082b08082b080808ULL, 0x082b08082b08082bULL, 0x082b0819082b2b19ULL, 0x082b081919082b08ULL, 0x082b082b08080808ULL, 0x082b082b0808082bULL, 0x082b190808080819ULL,
    0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b190819080808ULL, 0x082b19081919192bULL, 0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b1919192b1908ULL, 0x082b192b2b190808ULL,
    0x082b2b0808082b08ULL, 0x082b2b08082b0808ULL, 0x082b2b082b191908ULL, 0x082b2b2b19081908ULL, 0x1908080808080819ULL, 0x1908080808081908ULL, 0x1908080808190808ULL, 0x1908080808192b08ULL,
    0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x1908080819080808ULL, 0x1908080819082b08ULL, 0x190808081919192bULL, 0x19080808192b0808ULL, 0x190808082b080819ULL, 0x190808082b081908ULL,
    0x190808082b190808ULL, 0x1908081908080808ULL, 0x19080819082b0808ULL, 0x19080819192b0819ULL, 0x190808192b080808ULL, 0x190808192b081919ULL, 0x1908082b08080819ULL, 0x1908082b08190808ULL,
    0x1908082b19082b08ULL, 0x1908082b1919192bULL, 0x1908082b192b2b08ULL, 0x1908190808080808ULL, 0x1908190808082b08ULL, 0x19081908082b0808ULL, 0x190819082b080808ULL, 0x190819082b192b19ULL,
    0x190819190819082bULL, 0x19081919082b1908ULL, 0x1908192b08080808ULL, 0x19082b0808080819ULL, 0x19082b0808081908ULL, 0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL,
    0x19082b1908080808ULL, 0x19082b1919192b08ULL, 0x19082b19192b0819ULL, 0x19082b192b08082bULL, 0x19082b2b19081919ULL, 0x19082b2b2b190808ULL, 0x1919080808080808ULL, 0x1919080808082b08ULL,
    0x1919080808190819ULL, 0x1919080808192b19ULL, 0x19190808082b0808ULL, 0x191908082b080808ULL, 0x191908082b082b08ULL, 0x1919081908081908ULL, 0x191908191908082bULL, 0x191908192b2b1908ULL,
    0x1919082b2b190819ULL, 0x191919082b190808ULL, 0x191919082b19082bULL, 0x1919191908082b2bULL, 0x1919192b08080819ULL, 0x1919192b19191908ULL, 0x19192b0808080808ULL, 0x19192b0808190819ULL,
    0x19192b0808192b19ULL, 0x19192b08192b1908ULL, 0x19192b1919080808ULL, 0x19192b2b08082b08ULL, 0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL, 0x192b0808192b2b08ULL,
    0x192b081908080808ULL, 0x192b081919191919ULL, 0x192b082b08192b08ULL, 0x192b082b192b0808ULL, 0x192b190808080808ULL, 0x192b190808081919ULL, 0x192b191908190808ULL, 0x192b19190819082bULL,
    0x192b19192b081908ULL, 0x192b2b081908082bULL, 0x2b08080808080808ULL, 0x2b0808080808082bULL, 0x2b08080808082b2bULL, 0x2b08080819080819ULL, 0x2b0808082b08082bULL, 0x2b08081908081908ULL,
    0x2b08081908192b08ULL, 0x2b08081919080808ULL, 0x2b08082b08190819ULL, 0x2b08190808080819ULL, 0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b08190808191919ULL, 0x2b08190819080808ULL,
    0x2b081908192b0808ULL, 0x2b08191908080808ULL, 0x2b0819191908192bULL, 0x2b0819192b191908ULL, 0x2b08192b08082b19ULL, 0x2b08192b19080808ULL, 0x2b08192b192b0808ULL, 0x2b082b080808082bULL,
    0x2b082b1908081908ULL, 0x2b082b2b08190819ULL, 0x2b19080808081908ULL, 0x2b19080808190808ULL, 0x2b190808082b1908ULL, 0x2b19080819080808ULL, 0x2b1908082b2b0819ULL, 0x2b1908190819192bULL,
    0x2b1908192b080808ULL, 0x2b19082b19081919ULL, 0x2b19190808080808ULL, 0x2b191908082b082bULL, 0x2b19190819081908ULL, 0x2b19191919190819ULL, 0x2b192b082b080819ULL, 0x2b192b19082b0808ULL,
    0x2b2b08080808082bULL, 0x2b2b080819190808ULL, 0x2b2b08082b081919ULL, 0x2b2b081908082b19ULL, 0x2b2b082b08080808ULL, 0x2b2b190808192b08ULL, 0x2b2b2b0819190808ULL, 0x2b2b2b1908081908ULL,
};
__device__ __constant__ uint64_t IQ2XS_GRID_D[512] = {
    0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL, 0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x080808080819192bULL,
    0x0808080808192b19ULL, 0x08080808082b0808ULL, 0x08080808082b082bULL, 0x08080808082b1919ULL, 0x08080808082b2b08ULL, 0x0808080819080819ULL, 0x0808080819081908ULL, 0x080808081908192bULL,
    0x0808080819082b19ULL, 0x0808080819190808ULL, 0x080808081919082bULL, 0x0808080819191919ULL, 0x0808080819192b08ULL, 0x08080808192b0819ULL, 0x08080808192b1908ULL, 0x080808082b080808ULL,
    0x080808082b08082bULL, 0x080808082b081919ULL, 0x080808082b082b08ULL, 0x080808082b190819ULL, 0x080808082b191908ULL, 0x080808082b192b19ULL, 0x080808082b2b0808ULL, 0x0808081908080819ULL,
    0x0808081908081908ULL, 0x080808190808192bULL, 0x0808081908082b19ULL, 0x0808081908190808ULL, 0x080808190819082bULL, 0x0808081908191919ULL, 0x0808081908192b08ULL, 0x0808081908192b2bULL,
    0x08080819082b0819ULL, 0x08080819082b1908ULL, 0x0808081919080808ULL, 0x080808191908082bULL, 0x0808081919081919ULL, 0x0808081919082b08ULL, 0x0808081919190819ULL, 0x0808081919191908ULL,
    0x08080819192b0808ULL, 0x08080819192b2b08ULL, 0x080808192b080819ULL, 0x080808192b081908ULL, 0x080808192b190808ULL, 0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b08081919ULL,
    0x0808082b08082b08ULL, 0x0808082b08190819ULL, 0x0808082b08191908ULL, 0x0808082b082b0808ULL, 0x0808082b19080819ULL, 0x0808082b19081908ULL, 0x0808082b19190808ULL, 0x0808082b19191919ULL,
    0x0808082b2b080808ULL, 0x0808082b2b082b2bULL, 0x0808190808080819ULL, 0x0808190808081908ULL, 0x080819080808192bULL, 0x0808190808082b19ULL, 0x0808190808190808ULL, 0x080819080819082bULL,
    0x0808190808191919ULL, 0x0808190808192b08ULL, 0x08081908082b0819ULL, 0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819081919ULL, 0x0808190819082b08ULL,
    0x0808190819190819ULL, 0x0808190819191908ULL, 0x080819081919192bULL, 0x08081908192b0808ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL, 0x0808191908080808ULL,
    0x080819190808082bULL, 0x0808191908081919ULL, 0x0808191908082b08ULL, 0x0808191908190819ULL, 0x0808191908191908ULL, 0x08081919082b0808ULL, 0x0808191919080819ULL, 0x0808191919081908ULL,
    0x0808191919190808ULL, 0x08081919192b0819ULL, 0x080819192b080808ULL, 0x0808192b08080819ULL, 0x0808192b08081908ULL, 0x0808192b08190808ULL, 0x0808192b082b192bULL, 0x0808192b19080808ULL,
    0x0808192b1908082bULL, 0x0808192b2b081908ULL, 0x08082b0808080808ULL, 0x08082b080808082bULL, 0x08082b0808081919ULL, 0x08082b0808082b08ULL, 0x08082b0808082b2bULL, 0x08082b0808190819ULL,
    0x08082b0808191908ULL, 0x08082b08082b0808ULL, 0x08082b08082b1919ULL, 0x08082b0819080819ULL, 0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b0819192b08ULL, 0x08082b082b080808ULL,
    0x08082b082b2b0808ULL, 0x08082b082b2b2b2bULL, 0x08082b1908080819ULL, 0x08082b1908081908ULL, 0x08082b1908190808ULL, 0x08082b1919080808ULL, 0x08082b192b080819ULL, 0x08082b192b082b19ULL,
    0x08082b2b08080808ULL, 0x08082b2b082b0808ULL, 0x08082b2b082b2b08ULL, 0x08082b2b2b19192bULL, 0x08082b2b2b2b0808ULL, 0x0819080808080819ULL, 0x0819080808081908ULL, 0x081908080808192bULL,
    0x0819080808082b19ULL, 0x0819080808190808ULL, 0x081908080819082bULL, 0x0819080808191919ULL, 0x0819080808192b08ULL, 0x08190808082b0819ULL, 0x08190808082b1908ULL, 0x0819080819080808ULL,
    0x081908081908082bULL, 0x0819080819081919ULL, 0x0819080819082b08ULL, 0x0819080819190819ULL, 0x0819080819191908ULL, 0x08190808192b0808ULL, 0x08190808192b2b2bULL, 0x081908082b080819ULL,
    0x081908082b081908ULL, 0x081908082b190808ULL, 0x0819081908080808ULL, 0x081908190808082bULL, 0x0819081908081919ULL, 0x0819081908082b08ULL, 0x0819081908190819ULL, 0x0819081908191908ULL,
    0x08190819082b0808ULL, 0x0819081919080819ULL, 0x0819081919081908ULL, 0x0819081919190808ULL, 0x081908192b080808ULL, 0x081908192b191908ULL, 0x081908192b19192bULL, 0x0819082b08080819ULL,
    0x0819082b08081908ULL, 0x0819082b0808192bULL, 0x0819082b08190808ULL, 0x0819082b19080808ULL, 0x0819082b192b0808ULL, 0x0819190808080808ULL, 0x081919080808082bULL, 0x0819190808081919ULL,
    0x0819190808082b08ULL, 0x0819190808190819ULL, 0x0819190808191908ULL, 0x08191908082b0808ULL, 0x0819190819080819ULL, 0x0819190819081908ULL, 0x0819190819082b19ULL, 0x0819190819190808ULL,
    0x08191908192b1908ULL, 0x081919082b080808ULL, 0x0819191908080819ULL, 0x0819191908081908ULL, 0x0819191908190808ULL, 0x0819191919080808ULL, 0x0819192b08080808ULL, 0x0819192b08191908ULL,
    0x0819192b19082b19ULL, 0x08192b0808080819ULL, 0x08192b0808081908ULL, 0x08192b0808190808ULL, 0x08192b080819082bULL, 0x08192b0819080808ULL, 0x08192b0819191908ULL, 0x08192b082b08192bULL,
    0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b19192b192bULL, 0x08192b2b19190819ULL, 0x08192b2b2b2b2b19ULL, 0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808081919ULL,
    0x082b080808082b08ULL, 0x082b080808082b2bULL, 0x082b080808190819ULL, 0x082b080808191908ULL, 0x082b0808082b0808ULL, 0x082b080819080819ULL, 0x082b080819081908ULL, 0x082b080819190808ULL,
    0x082b08082b080808ULL, 0x082b08082b2b0808ULL, 0x082b081908080819ULL, 0x082b081908081908ULL, 0x082b081908190808ULL, 0x082b081919080808ULL, 0x082b081919082b08ULL, 0x082b0819192b1919ULL,
    0x082b082b08080808ULL, 0x082b082b082b082bULL, 0x082b082b2b080808ULL, 0x082b082b2b2b2b08ULL, 0x082b190808080819ULL, 0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b1908082b2b19ULL,
    0x082b190819080808ULL, 0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b19191919082bULL, 0x082b19192b192b19ULL, 0x082b192b08080819ULL, 0x082b192b08192b2bULL, 0x082b192b2b2b192bULL,
    0x082b2b0808080808ULL, 0x082b2b0808082b08ULL, 0x082b2b0808082b2bULL, 0x082b2b08082b0808ULL, 0x082b2b0819191919ULL, 0x082b2b082b082b08ULL, 0x082b2b082b2b082bULL, 0x082b2b19192b2b08ULL,
    0x082b2b192b190808ULL, 0x082b2b2b08082b08ULL, 0x082b2b2b082b0808ULL, 0x082b2b2b2b08082bULL, 0x082b2b2b2b082b08ULL, 0x082b2b2b2b082b2bULL, 0x1908080808080819ULL, 0x1908080808081908ULL,
    0x190808080808192bULL, 0x1908080808082b19ULL, 0x1908080808190808ULL, 0x190808080819082bULL, 0x1908080808191919ULL, 0x1908080808192b08ULL, 0x19080808082b0819ULL, 0x19080808082b1908ULL,
    0x1908080819080808ULL, 0x190808081908082bULL, 0x1908080819081919ULL, 0x1908080819082b08ULL, 0x1908080819082b2bULL, 0x1908080819190819ULL, 0x1908080819191908ULL, 0x19080808192b0808ULL,
    0x19080808192b1919ULL, 0x190808082b080819ULL, 0x190808082b081908ULL, 0x190808082b190808ULL, 0x1908081908080808ULL, 0x190808190808082bULL, 0x1908081908081919ULL, 0x1908081908082b08ULL,
    0x1908081908190819ULL, 0x1908081908191908ULL, 0x19080819082b0808ULL, 0x1908081919080819ULL, 0x1908081919081908ULL, 0x1908081919190808ULL, 0x190808192b080808ULL, 0x190808192b081919ULL,
    0x190808192b2b082bULL, 0x1908082b08080819ULL, 0x1908082b08081908ULL, 0x1908082b08190808ULL, 0x1908082b0819082bULL, 0x1908082b082b2b19ULL, 0x1908082b19080808ULL, 0x1908190808080808ULL,
    0x190819080808082bULL, 0x1908190808081919ULL, 0x1908190808082b08ULL, 0x1908190808190819ULL, 0x1908190808191908ULL, 0x1908190808192b19ULL, 0x19081908082b0808ULL, 0x1908190819080819ULL,
    0x1908190819081908ULL, 0x1908190819190808ULL, 0x190819082b080808ULL, 0x190819082b191908ULL, 0x1908191908080819ULL, 0x1908191908081908ULL, 0x1908191908190808ULL, 0x19081919082b1908ULL,
    0x1908191919080808ULL, 0x190819192b192b2bULL, 0x1908192b08080808ULL, 0x1908192b08082b2bULL, 0x1908192b19081908ULL, 0x1908192b19190808ULL, 0x19082b0808080819ULL, 0x19082b0808081908ULL,
    0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL, 0x19082b0819191908ULL, 0x19082b08192b082bULL, 0x19082b1908080808ULL, 0x19082b1908190819ULL, 0x19082b1919081908ULL,
    0x19082b1919190808ULL, 0x19082b19192b2b19ULL, 0x19082b2b08081908ULL, 0x1919080808080808ULL, 0x191908080808082bULL, 0x1919080808081919ULL, 0x1919080808082b08ULL, 0x1919080808190819ULL,
    0x1919080808191908ULL, 0x19190808082b0808ULL, 0x19190808082b2b08ULL, 0x1919080819080819ULL, 0x1919080819081908ULL, 0x1919080819190808ULL, 0x191908082b080808ULL, 0x1919081908080819ULL,
    0x1919081908081908ULL, 0x1919081908190808ULL, 0x1919081908191919ULL, 0x1919081919080808ULL, 0x191908191908082bULL, 0x1919082b08080808ULL, 0x1919082b19081908ULL, 0x1919082b2b2b2b2bULL,
    0x1919190808080819ULL, 0x1919190808081908ULL, 0x1919190808190808ULL, 0x19191908082b0819ULL, 0x1919190819080808ULL, 0x19191908192b0808ULL, 0x191919082b080819ULL, 0x191919082b2b0819ULL,
    0x1919191908080808ULL, 0x1919191908082b08ULL, 0x191919192b080808ULL, 0x191919192b082b08ULL, 0x1919192b082b0819ULL, 0x1919192b192b2b08ULL, 0x1919192b2b2b0819ULL, 0x19192b0808080808ULL,
    0x19192b0808191908ULL, 0x19192b0819080819ULL, 0x19192b0819190808ULL, 0x19192b082b192b19ULL, 0x19192b1908192b2bULL, 0x19192b1919080808ULL, 0x19192b191908082bULL, 0x19192b2b2b081919ULL,
    0x192b080808080819ULL, 0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL, 0x192b080819191908ULL, 0x192b0808192b082bULL, 0x192b08082b08192bULL, 0x192b08082b2b2b19ULL,
    0x192b081908080808ULL, 0x192b082b082b1908ULL, 0x192b082b19082b2bULL, 0x192b082b2b19082bULL, 0x192b190808080808ULL, 0x192b19080819192bULL, 0x192b191908190808ULL, 0x192b191919080808ULL,
    0x192b191919081919ULL, 0x192b19192b2b1908ULL, 0x192b2b0808080819ULL, 0x192b2b08192b2b2bULL, 0x192b2b19082b1919ULL, 0x192b2b2b0808192bULL, 0x192b2b2b19191908ULL, 0x192b2b2b192b082bULL,
    0x2b08080808080808ULL, 0x2b0808080808082bULL, 0x2b08080808081919ULL, 0x2b08080808082b08ULL, 0x2b08080808190819ULL, 0x2b08080808191908ULL, 0x2b080808082b0808ULL, 0x2b080808082b2b2bULL,
    0x2b08080819080819ULL, 0x2b08080819081908ULL, 0x2b08080819190808ULL, 0x2b0808082b080808ULL, 0x2b0808082b08082bULL, 0x2b0808082b2b2b08ULL, 0x2b0808082b2b2b2bULL, 0x2b08081908080819ULL,
    0x2b08081908081908ULL, 0x2b0808190808192bULL, 0x2b08081908190808ULL, 0x2b08081919080808ULL, 0x2b08081919190819ULL, 0x2b08081919192b19ULL, 0x2b08082b08080808ULL, 0x2b08082b082b0808ULL,
    0x2b08082b2b080808ULL, 0x2b08082b2b08082bULL, 0x2b08082b2b2b0808ULL, 0x2b08082b2b2b2b08ULL, 0x2b08190808080819ULL, 0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b0819080819082bULL,
    0x2b08190808191919ULL, 0x2b08190819080808ULL, 0x2b081908192b0808ULL, 0x2b0819082b082b19ULL, 0x2b08191908080808ULL, 0x2b08191919081908ULL, 0x2b0819192b2b1919ULL, 0x2b08192b08192b08ULL,
    0x2b08192b192b2b2bULL, 0x2b082b0808080808ULL, 0x2b082b0808082b08ULL, 0x2b082b08082b1919ULL, 0x2b082b0819192b2bULL, 0x2b082b082b080808ULL, 0x2b082b082b08082bULL, 0x2b082b082b2b2b08ULL,
    0x2b082b190808192bULL, 0x2b082b2b082b082bULL, 0x2b082b2b2b080808ULL, 0x2b082b2b2b082b08ULL, 0x2b082b2b2b19192bULL, 0x2b082b2b2b2b2b08ULL, 0x2b19080808080819ULL, 0x2b19080808081908ULL,
    0x2b19080808190808ULL, 0x2b19080819080808ULL, 0x2b1908081919192bULL, 0x2b1908082b081908ULL, 0x2b19081908080808ULL, 0x2b190819082b082bULL, 0x2b190819192b1908ULL, 0x2b19082b1919192bULL,
    0x2b19082b2b082b19ULL, 0x2b19190808080808ULL, 0x2b19190808081919ULL, 0x2b19190819081908ULL, 0x2b19190819190808ULL, 0x2b19190819192b08ULL, 0x2b191919082b2b19ULL, 0x2b1919192b190808ULL,
    0x2b1919192b19082bULL, 0x2b19192b19080819ULL, 0x2b192b0819190819ULL, 0x2b192b082b2b192bULL, 0x2b192b1919082b19ULL, 0x2b192b2b08191919ULL, 0x2b192b2b192b0808ULL, 0x2b2b080808080808ULL,
    0x2b2b08080808082bULL, 0x2b2b080808082b08ULL, 0x2b2b080808082b2bULL, 0x2b2b0808082b0808ULL, 0x2b2b0808082b2b2bULL, 0x2b2b08082b2b0808ULL, 0x2b2b081919190819ULL, 0x2b2b081919192b19ULL,
    0x2b2b08192b2b192bULL, 0x2b2b082b08080808ULL, 0x2b2b082b0808082bULL, 0x2b2b082b08082b08ULL, 0x2b2b082b082b2b2bULL, 0x2b2b082b2b080808ULL, 0x2b2b082b2b2b0808ULL, 0x2b2b190819080808ULL,
    0x2b2b19082b191919ULL, 0x2b2b192b192b1919ULL, 0x2b2b192b2b192b08ULL, 0x2b2b2b0808082b2bULL, 0x2b2b2b08082b0808ULL, 0x2b2b2b08082b082bULL, 0x2b2b2b08082b2b08ULL, 0x2b2b2b082b2b0808ULL,
    0x2b2b2b082b2b2b08ULL, 0x2b2b2b1908081908ULL, 0x2b2b2b192b081908ULL, 0x2b2b2b192b08192bULL, 0x2b2b2b2b082b2b08ULL, 0x2b2b2b2b082b2b2bULL, 0x2b2b2b2b2b190819ULL, 0x2b2b2b2b2b2b2b2bULL,
};
__device__ __constant__ uint64_t IQ2S_GRID_D[1024] = {
    0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL, 0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x080808080819192bULL,
    0x0808080808192b19ULL, 0x08080808082b0808ULL, 0x08080808082b082bULL, 0x08080808082b1919ULL, 0x08080808082b2b08ULL, 0x0808080819080819ULL, 0x0808080819081908ULL, 0x080808081908192bULL,
    0x0808080819082b19ULL, 0x0808080819190808ULL, 0x080808081919082bULL, 0x0808080819191919ULL, 0x0808080819192b08ULL, 0x08080808192b0819ULL, 0x08080808192b1908ULL, 0x08080808192b192bULL,
    0x08080808192b2b19ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b081919ULL, 0x080808082b082b08ULL, 0x080808082b190819ULL, 0x080808082b191908ULL, 0x080808082b2b0808ULL,
    0x080808082b2b1919ULL, 0x080808082b2b2b2bULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x080808190808192bULL, 0x0808081908082b19ULL, 0x0808081908190808ULL, 0x080808190819082bULL,
    0x0808081908191919ULL, 0x0808081908192b08ULL, 0x08080819082b0819ULL, 0x08080819082b1908ULL, 0x0808081919080808ULL, 0x080808191908082bULL, 0x0808081919081919ULL, 0x0808081919082b08ULL,
    0x0808081919190819ULL, 0x0808081919191908ULL, 0x080808191919192bULL, 0x0808081919192b19ULL, 0x08080819192b0808ULL, 0x08080819192b1919ULL, 0x08080819192b2b08ULL, 0x080808192b080819ULL,
    0x080808192b081908ULL, 0x080808192b190808ULL, 0x080808192b19082bULL, 0x080808192b191919ULL, 0x080808192b2b0819ULL, 0x080808192b2b1908ULL, 0x0808082b08080808ULL, 0x0808082b0808082bULL,
    0x0808082b08081919ULL, 0x0808082b08082b08ULL, 0x0808082b08190819ULL, 0x0808082b08191908ULL, 0x0808082b082b0808ULL, 0x0808082b082b2b2bULL, 0x0808082b19080819ULL, 0x0808082b19081908ULL,
    0x0808082b1908192bULL, 0x0808082b19082b19ULL, 0x0808082b19190808ULL, 0x0808082b19191919ULL, 0x0808082b2b080808ULL, 0x0808082b2b081919ULL, 0x0808082b2b082b2bULL, 0x0808082b2b191908ULL,
    0x0808082b2b2b082bULL, 0x0808190808080819ULL, 0x0808190808081908ULL, 0x080819080808192bULL, 0x0808190808082b19ULL, 0x0808190808190808ULL, 0x080819080819082bULL, 0x0808190808191919ULL,
    0x0808190808192b08ULL, 0x08081908082b0819ULL, 0x08081908082b1908ULL, 0x08081908082b192bULL, 0x08081908082b2b19ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819081919ULL,
    0x0808190819082b08ULL, 0x0808190819082b2bULL, 0x0808190819190819ULL, 0x0808190819191908ULL, 0x080819081919192bULL, 0x0808190819192b19ULL, 0x08081908192b0808ULL, 0x08081908192b082bULL,
    0x08081908192b1919ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b08192bULL, 0x080819082b082b19ULL, 0x080819082b190808ULL, 0x080819082b191919ULL, 0x080819082b192b08ULL,
    0x080819082b2b0819ULL, 0x080819082b2b1908ULL, 0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908081919ULL, 0x0808191908082b08ULL, 0x0808191908082b2bULL, 0x0808191908190819ULL,
    0x0808191908191908ULL, 0x080819190819192bULL, 0x0808191908192b19ULL, 0x08081919082b0808ULL, 0x08081919082b1919ULL, 0x08081919082b2b08ULL, 0x0808191919080819ULL, 0x0808191919081908ULL,
    0x080819191908192bULL, 0x0808191919082b19ULL, 0x0808191919190808ULL, 0x080819191919082bULL, 0x0808191919191919ULL, 0x0808191919192b08ULL, 0x08081919192b0819ULL, 0x08081919192b1908ULL,
    0x080819192b080808ULL, 0x080819192b08082bULL, 0x080819192b081919ULL, 0x080819192b082b08ULL, 0x080819192b190819ULL, 0x080819192b191908ULL, 0x080819192b2b0808ULL, 0x0808192b08080819ULL,
    0x0808192b08081908ULL, 0x0808192b0808192bULL, 0x0808192b08082b19ULL, 0x0808192b08190808ULL, 0x0808192b08191919ULL, 0x0808192b19080808ULL, 0x0808192b19081919ULL, 0x0808192b19082b08ULL,
    0x0808192b19190819ULL, 0x0808192b19191908ULL, 0x0808192b192b0808ULL, 0x0808192b2b080819ULL, 0x0808192b2b081908ULL, 0x0808192b2b190808ULL, 0x08082b0808080808ULL, 0x08082b080808082bULL,
    0x08082b0808081919ULL, 0x08082b0808082b08ULL, 0x08082b0808190819ULL, 0x08082b0808191908ULL, 0x08082b080819192bULL, 0x08082b0808192b19ULL, 0x08082b08082b0808ULL, 0x08082b08082b1919ULL,
    0x08082b08082b2b2bULL, 0x08082b0819080819ULL, 0x08082b0819081908ULL, 0x08082b081908192bULL, 0x08082b0819082b19ULL, 0x08082b0819190808ULL, 0x08082b081919082bULL, 0x08082b0819191919ULL,
    0x08082b0819192b08ULL, 0x08082b08192b0819ULL, 0x08082b08192b1908ULL, 0x08082b082b080808ULL, 0x08082b082b081919ULL, 0x08082b082b191908ULL, 0x08082b082b2b2b2bULL, 0x08082b1908080819ULL,
    0x08082b1908081908ULL, 0x08082b1908190808ULL, 0x08082b190819082bULL, 0x08082b1908191919ULL, 0x08082b1908192b08ULL, 0x08082b19082b0819ULL, 0x08082b1919080808ULL, 0x08082b1919081919ULL,
    0x08082b1919082b08ULL, 0x08082b1919190819ULL, 0x08082b1919191908ULL, 0x08082b19192b0808ULL, 0x08082b192b080819ULL, 0x08082b192b190808ULL, 0x08082b2b08080808ULL, 0x08082b2b08190819ULL,
    0x08082b2b08191908ULL, 0x08082b2b082b082bULL, 0x08082b2b082b2b08ULL, 0x08082b2b082b2b2bULL, 0x08082b2b19190808ULL, 0x08082b2b2b192b19ULL, 0x0819080808080819ULL, 0x0819080808081908ULL,
    0x081908080808192bULL, 0x0819080808082b19ULL, 0x0819080808190808ULL, 0x081908080819082bULL, 0x0819080808191919ULL, 0x0819080808192b08ULL, 0x08190808082b0819ULL, 0x08190808082b1908ULL,
    0x08190808082b192bULL, 0x0819080819080808ULL, 0x081908081908082bULL, 0x0819080819081919ULL, 0x0819080819082b08ULL, 0x0819080819190819ULL, 0x0819080819191908ULL, 0x081908081919192bULL,
    0x0819080819192b19ULL, 0x08190808192b0808ULL, 0x08190808192b082bULL, 0x08190808192b1919ULL, 0x08190808192b2b08ULL, 0x081908082b080819ULL, 0x081908082b081908ULL, 0x081908082b08192bULL,
    0x081908082b190808ULL, 0x081908082b191919ULL, 0x081908082b192b08ULL, 0x081908082b2b0819ULL, 0x081908082b2b1908ULL, 0x0819081908080808ULL, 0x081908190808082bULL, 0x0819081908081919ULL,
    0x0819081908082b08ULL, 0x0819081908082b2bULL, 0x0819081908190819ULL, 0x0819081908191908ULL, 0x081908190819192bULL, 0x0819081908192b19ULL, 0x08190819082b0808ULL, 0x08190819082b082bULL,
    0x08190819082b1919ULL, 0x08190819082b2b08ULL, 0x0819081919080819ULL, 0x0819081919081908ULL, 0x081908191908192bULL, 0x0819081919082b19ULL, 0x0819081919190808ULL, 0x081908191919082bULL,
    0x0819081919191919ULL, 0x0819081919192b08ULL, 0x08190819192b0819ULL, 0x08190819192b1908ULL, 0x081908192b080808ULL, 0x081908192b08082bULL, 0x081908192b081919ULL, 0x081908192b082b08ULL,
    0x081908192b190819ULL, 0x081908192b191908ULL, 0x0819082b08080819ULL, 0x0819082b08081908ULL, 0x0819082b08082b19ULL, 0x0819082b08190808ULL, 0x0819082b08191919ULL, 0x0819082b082b0819ULL,
    0x0819082b082b1908ULL, 0x0819082b19080808ULL, 0x0819082b19081919ULL, 0x0819082b19190819ULL, 0x0819082b19191908ULL, 0x0819082b2b080819ULL, 0x0819082b2b081908ULL, 0x0819082b2b190808ULL,
    0x0819190808080808ULL, 0x081919080808082bULL, 0x0819190808081919ULL, 0x0819190808082b08ULL, 0x0819190808190819ULL, 0x0819190808191908ULL, 0x081919080819192bULL, 0x0819190808192b19ULL,
    0x08191908082b0808ULL, 0x08191908082b1919ULL, 0x08191908082b2b08ULL, 0x0819190819080819ULL, 0x0819190819081908ULL, 0x081919081908192bULL, 0x0819190819082b19ULL, 0x0819190819190808ULL,
    0x081919081919082bULL, 0x0819190819191919ULL, 0x0819190819192b08ULL, 0x08191908192b0819ULL, 0x08191908192b1908ULL, 0x081919082b080808ULL, 0x081919082b08082bULL, 0x081919082b081919ULL,
    0x081919082b082b08ULL, 0x081919082b190819ULL, 0x081919082b191908ULL, 0x081919082b2b0808ULL, 0x0819191908080819ULL, 0x0819191908081908ULL, 0x081919190808192bULL, 0x0819191908082b19ULL,
    0x0819191908190808ULL, 0x081919190819082bULL, 0x0819191908191919ULL, 0x0819191908192b08ULL, 0x08191919082b0819ULL, 0x08191919082b1908ULL, 0x0819191919080808ULL, 0x081919191908082bULL,
    0x0819191919081919ULL, 0x0819191919082b08ULL, 0x0819191919190819ULL, 0x0819191919191908ULL, 0x08191919192b0808ULL, 0x081919192b080819ULL, 0x081919192b081908ULL, 0x081919192b190808ULL,
    0x0819192b08080808ULL, 0x0819192b08081919ULL, 0x0819192b08082b08ULL, 0x0819192b08190819ULL, 0x0819192b08191908ULL, 0x0819192b082b0808ULL, 0x0819192b19080819ULL, 0x0819192b19081908ULL,
    0x0819192b19190808ULL, 0x0819192b2b080808ULL, 0x0819192b2b2b2b2bULL, 0x08192b0808080819ULL, 0x08192b0808081908ULL, 0x08192b080808192bULL, 0x08192b0808082b19ULL, 0x08192b0808190808ULL,
    0x08192b0808191919ULL, 0x08192b0808192b08ULL, 0x08192b08082b0819ULL, 0x08192b0819080808ULL, 0x08192b081908082bULL, 0x08192b0819081919ULL, 0x08192b0819082b08ULL, 0x08192b0819190819ULL,
    0x08192b0819191908ULL, 0x08192b08192b0808ULL, 0x08192b082b080819ULL, 0x08192b082b081908ULL, 0x08192b1908080808ULL, 0x08192b190808082bULL, 0x08192b1908081919ULL, 0x08192b1908082b08ULL,
    0x08192b1908190819ULL, 0x08192b1908191908ULL, 0x08192b19082b0808ULL, 0x08192b1919080819ULL, 0x08192b1919081908ULL, 0x08192b1919190808ULL, 0x08192b19192b2b19ULL, 0x08192b192b2b082bULL,
    0x08192b2b08081908ULL, 0x08192b2b08190808ULL, 0x08192b2b19080808ULL, 0x08192b2b1919192bULL, 0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808081919ULL, 0x082b080808082b08ULL,
    0x082b080808190819ULL, 0x082b080808191908ULL, 0x082b08080819192bULL, 0x082b080808192b19ULL, 0x082b0808082b0808ULL, 0x082b0808082b1919ULL, 0x082b0808082b2b2bULL, 0x082b080819080819ULL,
    0x082b080819081908ULL, 0x082b080819190808ULL, 0x082b08081919082bULL, 0x082b080819191919ULL, 0x082b0808192b1908ULL, 0x082b08082b080808ULL, 0x082b08082b082b2bULL, 0x082b08082b191908ULL,
    0x082b08082b2b2b2bULL, 0x082b081908080819ULL, 0x082b081908081908ULL, 0x082b081908190808ULL, 0x082b08190819082bULL, 0x082b081908191919ULL, 0x082b0819082b0819ULL, 0x082b081919080808ULL,
    0x082b08191908082bULL, 0x082b081919081919ULL, 0x082b081919190819ULL, 0x082b081919191908ULL, 0x082b0819192b0808ULL, 0x082b08192b080819ULL, 0x082b08192b081908ULL, 0x082b08192b190808ULL,
    0x082b082b08080808ULL, 0x082b082b08082b2bULL, 0x082b082b082b082bULL, 0x082b082b082b2b08ULL, 0x082b082b082b2b2bULL, 0x082b082b19081908ULL, 0x082b082b19190808ULL, 0x082b082b2b082b08ULL,
    0x082b082b2b082b2bULL, 0x082b082b2b2b2b08ULL, 0x082b190808080819ULL, 0x082b190808081908ULL, 0x082b19080808192bULL, 0x082b190808082b19ULL, 0x082b190808190808ULL, 0x082b190808191919ULL,
    0x082b190808192b08ULL, 0x082b1908082b0819ULL, 0x082b1908082b1908ULL, 0x082b190819080808ULL, 0x082b19081908082bULL, 0x082b190819081919ULL, 0x082b190819082b08ULL, 0x082b190819190819ULL,
    0x082b190819191908ULL, 0x082b1908192b0808ULL, 0x082b19082b080819ULL, 0x082b19082b081908ULL, 0x082b19082b190808ULL, 0x082b191908080808ULL, 0x082b191908081919ULL, 0x082b191908082b08ULL,
    0x082b191908190819ULL, 0x082b191908191908ULL, 0x082b1919082b0808ULL, 0x082b191919080819ULL, 0x082b191919081908ULL, 0x082b191919190808ULL, 0x082b1919192b192bULL, 0x082b19192b080808ULL,
    0x082b192b08080819ULL, 0x082b192b08081908ULL, 0x082b192b08190808ULL, 0x082b192b19080808ULL, 0x082b192b19192b19ULL, 0x082b2b0808080808ULL, 0x082b2b0808081919ULL, 0x082b2b0808190819ULL,
    0x082b2b0808191908ULL, 0x082b2b0819080819ULL, 0x082b2b0819081908ULL, 0x082b2b0819190808ULL, 0x082b2b082b082b2bULL, 0x082b2b082b2b2b2bULL, 0x082b2b1908080819ULL, 0x082b2b1908081908ULL,
    0x082b2b1908190808ULL, 0x082b2b192b191919ULL, 0x082b2b2b08082b2bULL, 0x082b2b2b082b082bULL, 0x082b2b2b192b1908ULL, 0x082b2b2b2b082b08ULL, 0x082b2b2b2b082b2bULL, 0x1908080808080819ULL,
    0x1908080808081908ULL, 0x190808080808192bULL, 0x1908080808082b19ULL, 0x1908080808190808ULL, 0x190808080819082bULL, 0x1908080808191919ULL, 0x1908080808192b08ULL, 0x1908080808192b2bULL,
    0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x19080808082b192bULL, 0x1908080819080808ULL, 0x190808081908082bULL, 0x1908080819081919ULL, 0x1908080819082b08ULL, 0x1908080819082b2bULL,
    0x1908080819190819ULL, 0x1908080819191908ULL, 0x190808081919192bULL, 0x1908080819192b19ULL, 0x19080808192b0808ULL, 0x19080808192b082bULL, 0x19080808192b1919ULL, 0x190808082b080819ULL,
    0x190808082b081908ULL, 0x190808082b190808ULL, 0x190808082b191919ULL, 0x190808082b192b08ULL, 0x190808082b2b0819ULL, 0x190808082b2b1908ULL, 0x1908081908080808ULL, 0x190808190808082bULL,
    0x1908081908081919ULL, 0x1908081908082b08ULL, 0x1908081908190819ULL, 0x1908081908191908ULL, 0x190808190819192bULL, 0x1908081908192b19ULL, 0x19080819082b0808ULL, 0x19080819082b082bULL,
    0x19080819082b1919ULL, 0x1908081919080819ULL, 0x1908081919081908ULL, 0x190808191908192bULL, 0x1908081919082b19ULL, 0x1908081919190808ULL, 0x190808191919082bULL, 0x1908081919191919ULL,
    0x1908081919192b08ULL, 0x19080819192b0819ULL, 0x19080819192b1908ULL, 0x190808192b080808ULL, 0x190808192b08082bULL, 0x190808192b081919ULL, 0x190808192b082b08ULL, 0x190808192b190819ULL,
    0x190808192b191908ULL, 0x190808192b2b0808ULL, 0x1908082b08080819ULL, 0x1908082b08081908ULL, 0x1908082b08190808ULL, 0x1908082b0819082bULL, 0x1908082b08191919ULL, 0x1908082b08192b08ULL,
    0x1908082b082b1908ULL, 0x1908082b19080808ULL, 0x1908082b19081919ULL, 0x1908082b19082b08ULL, 0x1908082b19190819ULL, 0x1908082b19191908ULL, 0x1908082b192b0808ULL, 0x1908082b2b080819ULL,
    0x1908082b2b081908ULL, 0x1908190808080808ULL, 0x190819080808082bULL, 0x1908190808081919ULL, 0x1908190808082b08ULL, 0x1908190808082b2bULL, 0x1908190808190819ULL, 0x1908190808191908ULL,
    0x190819080819192bULL, 0x1908190808192b19ULL, 0x19081908082b0808ULL, 0x19081908082b082bULL, 0x19081908082b1919ULL, 0x19081908082b2b08ULL, 0x1908190819080819ULL, 0x1908190819081908ULL,
    0x190819081908192bULL, 0x1908190819082b19ULL, 0x1908190819190808ULL, 0x190819081919082bULL, 0x1908190819191919ULL, 0x1908190819192b08ULL, 0x19081908192b0819ULL, 0x19081908192b1908ULL,
    0x190819082b080808ULL, 0x190819082b08082bULL, 0x190819082b081919ULL, 0x190819082b082b08ULL, 0x190819082b190819ULL, 0x190819082b191908ULL, 0x190819082b2b0808ULL, 0x1908191908080819ULL,
    0x1908191908081908ULL, 0x190819190808192bULL, 0x1908191908082b19ULL, 0x1908191908190808ULL, 0x190819190819082bULL, 0x1908191908191919ULL, 0x1908191908192b08ULL, 0x19081919082b0819ULL,
    0x19081919082b1908ULL, 0x1908191919080808ULL, 0x190819191908082bULL, 0x1908191919081919ULL, 0x1908191919082b08ULL, 0x1908191919190819ULL, 0x1908191919191908ULL, 0x19081919192b0808ULL,
    0x19081919192b2b2bULL, 0x190819192b080819ULL, 0x190819192b081908ULL, 0x190819192b190808ULL, 0x1908192b08080808ULL, 0x1908192b0808082bULL, 0x1908192b08081919ULL, 0x1908192b08082b08ULL,
    0x1908192b08190819ULL, 0x1908192b08191908ULL, 0x1908192b082b0808ULL, 0x1908192b19080819ULL, 0x1908192b19081908ULL, 0x1908192b19190808ULL, 0x1908192b2b080808ULL, 0x1908192b2b2b1919ULL,
    0x19082b0808080819ULL, 0x19082b0808081908ULL, 0x19082b0808082b19ULL, 0x19082b0808190808ULL, 0x19082b080819082bULL, 0x19082b0808191919ULL, 0x19082b0808192b08ULL, 0x19082b08082b0819ULL,
    0x19082b08082b1908ULL, 0x19082b0819080808ULL, 0x19082b081908082bULL, 0x19082b0819081919ULL, 0x19082b0819082b08ULL, 0x19082b0819190819ULL, 0x19082b0819191908ULL, 0x19082b08192b0808ULL,
    0x19082b082b081908ULL, 0x19082b082b190808ULL, 0x19082b1908080808ULL, 0x19082b190808082bULL, 0x19082b1908081919ULL, 0x19082b1908082b08ULL, 0x19082b1908190819ULL, 0x19082b1908191908ULL,
    0x19082b19082b0808ULL, 0x19082b1919080819ULL, 0x19082b1919081908ULL, 0x19082b1919190808ULL, 0x19082b192b080808ULL, 0x19082b192b19192bULL, 0x19082b2b08080819ULL, 0x19082b2b08081908ULL,
    0x19082b2b08190808ULL, 0x19082b2b19080808ULL, 0x1919080808080808ULL, 0x191908080808082bULL, 0x1919080808081919ULL, 0x1919080808082b08ULL, 0x1919080808190819ULL, 0x1919080808191908ULL,
    0x191908080819192bULL, 0x1919080808192b19ULL, 0x19190808082b0808ULL, 0x19190808082b082bULL, 0x19190808082b1919ULL, 0x19190808082b2b08ULL, 0x1919080819080819ULL, 0x1919080819081908ULL,
    0x191908081908192bULL, 0x1919080819082b19ULL, 0x1919080819190808ULL, 0x191908081919082bULL, 0x1919080819191919ULL, 0x1919080819192b08ULL, 0x19190808192b0819ULL, 0x19190808192b1908ULL,
    0x191908082b080808ULL, 0x191908082b08082bULL, 0x191908082b081919ULL, 0x191908082b082b08ULL, 0x191908082b190819ULL, 0x191908082b191908ULL, 0x1919081908080819ULL, 0x1919081908081908ULL,
    0x191908190808192bULL, 0x1919081908082b19ULL, 0x1919081908190808ULL, 0x191908190819082bULL, 0x1919081908191919ULL, 0x1919081908192b08ULL, 0x19190819082b0819ULL, 0x19190819082b1908ULL,
    0x1919081919080808ULL, 0x191908191908082bULL, 0x1919081919081919ULL, 0x1919081919082b08ULL, 0x1919081919190819ULL, 0x1919081919191908ULL, 0x19190819192b0808ULL, 0x191908192b080819ULL,
    0x191908192b081908ULL, 0x191908192b190808ULL, 0x1919082b08080808ULL, 0x1919082b08081919ULL, 0x1919082b08082b08ULL, 0x1919082b08190819ULL, 0x1919082b08191908ULL, 0x1919082b082b0808ULL,
    0x1919082b19080819ULL, 0x1919082b19081908ULL, 0x1919082b19190808ULL, 0x1919082b192b2b19ULL, 0x1919082b2b080808ULL, 0x1919190808080819ULL, 0x1919190808081908ULL, 0x191919080808192bULL,
    0x1919190808082b19ULL, 0x1919190808190808ULL, 0x191919080819082bULL, 0x1919190808191919ULL, 0x1919190808192b08ULL, 0x19191908082b0819ULL, 0x19191908082b1908ULL, 0x1919190819080808ULL,
    0x191919081908082bULL, 0x1919190819081919ULL, 0x1919190819082b08ULL, 0x1919190819190819ULL, 0x1919190819191908ULL, 0x19191908192b0808ULL, 0x191919082b080819ULL, 0x191919082b081908ULL,
    0x191919082b190808ULL, 0x1919191908080808ULL, 0x191919190808082bULL, 0x1919191908081919ULL, 0x1919191908082b08ULL, 0x1919191908190819ULL, 0x1919191908191908ULL, 0x19191919082b0808ULL,
    0x1919191919080819ULL, 0x1919191919081908ULL, 0x1919191919190808ULL, 0x191919192b080808ULL, 0x1919192b08080819ULL, 0x1919192b08081908ULL, 0x1919192b08190808ULL, 0x1919192b082b192bULL,
    0x1919192b19080808ULL, 0x19192b0808080808ULL, 0x19192b080808082bULL, 0x19192b0808081919ULL, 0x19192b0808082b08ULL, 0x19192b0808190819ULL, 0x19192b0808191908ULL, 0x19192b08082b0808ULL,
    0x19192b0819080819ULL, 0x19192b0819081908ULL, 0x19192b0819190808ULL, 0x19192b0819192b2bULL, 0x19192b082b080808ULL, 0x19192b1908080819ULL, 0x19192b1908081908ULL, 0x19192b1908190808ULL,
    0x19192b1919080808ULL, 0x19192b2b08080808ULL, 0x19192b2b08192b19ULL, 0x19192b2b2b081919ULL, 0x19192b2b2b2b2b08ULL, 0x192b080808080819ULL, 0x192b080808081908ULL, 0x192b08080808192bULL,
    0x192b080808190808ULL, 0x192b08080819082bULL, 0x192b080808191919ULL, 0x192b080808192b08ULL, 0x192b0808082b0819ULL, 0x192b0808082b1908ULL, 0x192b080819080808ULL, 0x192b080819081919ULL,
    0x192b080819082b08ULL, 0x192b080819190819ULL, 0x192b080819191908ULL, 0x192b0808192b0808ULL, 0x192b08082b081908ULL, 0x192b08082b190808ULL, 0x192b081908080808ULL, 0x192b08190808082bULL,
    0x192b081908081919ULL, 0x192b081908082b08ULL, 0x192b081908190819ULL, 0x192b081908191908ULL, 0x192b0819082b0808ULL, 0x192b081919080819ULL, 0x192b081919081908ULL, 0x192b081919190808ULL,
    0x192b08192b080808ULL, 0x192b08192b192b19ULL, 0x192b082b08081908ULL, 0x192b082b08190808ULL, 0x192b082b19080808ULL, 0x192b082b1919192bULL, 0x192b082b2b2b0819ULL, 0x192b190808080808ULL,
    0x192b190808081919ULL, 0x192b190808082b08ULL, 0x192b190808190819ULL, 0x192b190808191908ULL, 0x192b1908082b0808ULL, 0x192b190819080819ULL, 0x192b190819081908ULL, 0x192b190819190808ULL,
    0x192b19082b080808ULL, 0x192b191908080819ULL, 0x192b191908081908ULL, 0x192b191908190808ULL, 0x192b191919080808ULL, 0x192b191919082b2bULL, 0x192b1919192b2b08ULL, 0x192b19192b19082bULL,
    0x192b192b08080808ULL, 0x192b192b2b191908ULL, 0x192b2b0808080819ULL, 0x192b2b0808081908ULL, 0x192b2b0808190808ULL, 0x192b2b08192b1919ULL, 0x192b2b082b192b08ULL, 0x192b2b1908080808ULL,
    0x192b2b19082b2b2bULL, 0x192b2b2b1908082bULL, 0x192b2b2b2b2b0819ULL, 0x2b08080808080808ULL, 0x2b0808080808082bULL, 0x2b08080808081919ULL, 0x2b08080808082b08ULL, 0x2b08080808190819ULL,
    0x2b08080808191908ULL, 0x2b08080808192b19ULL, 0x2b080808082b0808ULL, 0x2b080808082b1919ULL, 0x2b08080819080819ULL, 0x2b08080819081908ULL, 0x2b08080819190808ULL, 0x2b0808081919082bULL,
    0x2b08080819191919ULL, 0x2b08080819192b08ULL, 0x2b080808192b0819ULL, 0x2b0808082b080808ULL, 0x2b0808082b081919ULL, 0x2b0808082b190819ULL, 0x2b0808082b191908ULL, 0x2b08081908080819ULL,
    0x2b08081908081908ULL, 0x2b08081908082b19ULL, 0x2b08081908190808ULL, 0x2b0808190819082bULL, 0x2b08081908191919ULL, 0x2b08081908192b08ULL, 0x2b080819082b0819ULL, 0x2b080819082b1908ULL,
    0x2b08081919080808ULL, 0x2b0808191908082bULL, 0x2b08081919081919ULL, 0x2b08081919082b08ULL, 0x2b08081919190819ULL, 0x2b08081919191908ULL, 0x2b0808192b080819ULL, 0x2b0808192b081908ULL,
    0x2b0808192b190808ULL, 0x2b0808192b2b2b19ULL, 0x2b08082b08080808ULL, 0x2b08082b08081919ULL, 0x2b08082b08082b2bULL, 0x2b08082b08190819ULL, 0x2b08082b08191908ULL, 0x2b08082b19080819ULL,
    0x2b08082b19081908ULL, 0x2b08082b19190808ULL, 0x2b08190808080819ULL, 0x2b08190808081908ULL, 0x2b0819080808192bULL, 0x2b08190808082b19ULL, 0x2b08190808190808ULL, 0x2b0819080819082bULL,
    0x2b08190808191919ULL, 0x2b08190808192b08ULL, 0x2b081908082b0819ULL, 0x2b08190819080808ULL, 0x2b0819081908082bULL, 0x2b08190819081919ULL, 0x2b08190819082b08ULL, 0x2b08190819190819ULL,
    0x2b08190819191908ULL, 0x2b081908192b0808ULL, 0x2b0819082b080819ULL, 0x2b0819082b081908ULL, 0x2b0819082b190808ULL, 0x2b08191908080808ULL, 0x2b0819190808082bULL, 0x2b08191908081919ULL,
    0x2b08191908082b08ULL, 0x2b08191908190819ULL, 0x2b08191908191908ULL, 0x2b081919082b0808ULL, 0x2b08191919080819ULL, 0x2b08191919081908ULL, 0x2b08191919190808ULL, 0x2b0819192b080808ULL,
    0x2b0819192b082b2bULL, 0x2b08192b08080819ULL, 0x2b08192b08081908ULL, 0x2b08192b08190808ULL, 0x2b08192b082b2b19ULL, 0x2b08192b19080808ULL, 0x2b082b0808080808ULL, 0x2b082b0808081919ULL,
    0x2b082b0808190819ULL, 0x2b082b0808191908ULL, 0x2b082b0819080819ULL, 0x2b082b0819081908ULL, 0x2b082b0819190808ULL, 0x2b082b082b2b082bULL, 0x2b082b1908080819ULL, 0x2b082b1908081908ULL,
    0x2b082b1919080808ULL, 0x2b082b19192b1919ULL, 0x2b082b2b082b082bULL, 0x2b082b2b19192b08ULL, 0x2b082b2b19192b2bULL, 0x2b082b2b2b08082bULL, 0x2b082b2b2b2b082bULL, 0x2b19080808080819ULL,
    0x2b19080808081908ULL, 0x2b19080808082b19ULL, 0x2b19080808190808ULL, 0x2b1908080819082bULL, 0x2b19080808191919ULL, 0x2b19080808192b08ULL, 0x2b190808082b1908ULL, 0x2b19080819080808ULL,
    0x2b1908081908082bULL, 0x2b19080819081919ULL, 0x2b19080819082b08ULL, 0x2b19080819190819ULL, 0x2b19080819191908ULL, 0x2b190808192b0808ULL, 0x2b1908082b080819ULL, 0x2b1908082b081908ULL,
    0x2b1908082b190808ULL, 0x2b19081908080808ULL, 0x2b19081908081919ULL, 0x2b19081908190819ULL, 0x2b19081908191908ULL, 0x2b19081919080819ULL, 0x2b19081919081908ULL, 0x2b19081919190808ULL,
    0x2b19081919192b2bULL, 0x2b19082b08080819ULL, 0x2b19082b08081908ULL, 0x2b19082b08190808ULL, 0x2b19082b19080808ULL, 0x2b19082b2b2b192bULL, 0x2b19190808080808ULL, 0x2b1919080808082bULL,
    0x2b19190808081919ULL, 0x2b19190808082b08ULL, 0x2b19190808190819ULL, 0x2b19190808191908ULL, 0x2b191908082b0808ULL, 0x2b19190819080819ULL, 0x2b19190819081908ULL, 0x2b19190819190808ULL,
    0x2b1919082b080808ULL, 0x2b1919082b19192bULL, 0x2b19191908080819ULL, 0x2b19191908081908ULL, 0x2b19191908190808ULL, 0x2b19191919080808ULL, 0x2b1919192b192b08ULL, 0x2b1919192b2b0819ULL,
    0x2b19192b08080808ULL, 0x2b19192b1908192bULL, 0x2b19192b192b1908ULL, 0x2b192b0808080819ULL, 0x2b192b0808081908ULL, 0x2b192b0808190808ULL, 0x2b192b08082b192bULL, 0x2b192b0819080808ULL,
    0x2b192b082b2b2b19ULL, 0x2b192b1908080808ULL, 0x2b192b1919082b19ULL, 0x2b192b191919082bULL, 0x2b192b2b2b190808ULL, 0x2b2b080808080808ULL, 0x2b2b080808081919ULL, 0x2b2b080808082b2bULL,
    0x2b2b080808191908ULL, 0x2b2b0808082b082bULL, 0x2b2b0808082b2b2bULL, 0x2b2b080819080819ULL, 0x2b2b080819081908ULL, 0x2b2b080819190808ULL, 0x2b2b08082b2b082bULL, 0x2b2b08082b2b2b2bULL,
    0x2b2b081919080808ULL, 0x2b2b0819192b1919ULL, 0x2b2b082b0808082bULL, 0x2b2b082b08082b2bULL, 0x2b2b082b082b082bULL, 0x2b2b082b082b2b08ULL, 0x2b2b082b082b2b2bULL, 0x2b2b082b2b08082bULL,
    0x2b2b082b2b082b08ULL, 0x2b2b082b2b082b2bULL, 0x2b2b082b2b2b2b08ULL, 0x2b2b190808080819ULL, 0x2b2b190808081908ULL, 0x2b2b190808190808ULL, 0x2b2b190819080808ULL, 0x2b2b19082b082b19ULL,
    0x2b2b19082b2b1908ULL, 0x2b2b191908080808ULL, 0x2b2b191908192b19ULL, 0x2b2b192b19190819ULL, 0x2b2b2b0808082b2bULL, 0x2b2b2b08082b2b08ULL, 0x2b2b2b082b2b082bULL, 0x2b2b2b1919191908ULL,
    0x2b2b2b192b08192bULL, 0x2b2b2b2b08082b08ULL, 0x2b2b2b2b08082b2bULL, 0x2b2b2b2b082b0808ULL, 0x2b2b2b2b082b082bULL, 0x2b2b2b2b082b2b08ULL, 0x2b2b2b2b2b082b08ULL, 0x2b2b2b2b2b2b2b2bULL,
};
__device__ __constant__ uint32_t IQ3XXS_GRID_D[256] = {
    0x04040404u, 0x04040414u, 0x04040424u, 0x04040c0cu, 0x04040c1cu, 0x04040c3eu, 0x04041404u, 0x04041414u,
    0x04041c0cu, 0x04042414u, 0x04043e1cu, 0x04043e2cu, 0x040c040cu, 0x040c041cu, 0x040c0c04u, 0x040c0c14u,
    0x040c140cu, 0x040c142cu, 0x040c1c04u, 0x040c1c14u, 0x040c240cu, 0x040c2c24u, 0x040c3e04u, 0x04140404u,
    0x04140414u, 0x04140424u, 0x04140c0cu, 0x04141404u, 0x04141414u, 0x04141c0cu, 0x04141c1cu, 0x04141c3eu,
    0x04142c0cu, 0x04142c3eu, 0x04143e2cu, 0x041c040cu, 0x041c043eu, 0x041c0c04u, 0x041c0c14u, 0x041c142cu,
    0x041c3e04u, 0x04240c1cu, 0x04241c3eu, 0x04242424u, 0x04242c3eu, 0x04243e1cu, 0x04243e2cu, 0x042c040cu,
    0x042c043eu, 0x042c1c14u, 0x042c2c14u, 0x04341c2cu, 0x04343424u, 0x043e0c04u, 0x043e0c24u, 0x043e0c34u,
    0x043e241cu, 0x043e340cu, 0x0c04040cu, 0x0c04041cu, 0x0c040c04u, 0x0c040c14u, 0x0c04140cu, 0x0c04141cu,
    0x0c041c04u, 0x0c041c14u, 0x0c041c24u, 0x0c04243eu, 0x0c042c04u, 0x0c0c0404u, 0x0c0c0414u, 0x0c0c0c0cu,
    0x0c0c1404u, 0x0c0c1414u, 0x0c14040cu, 0x0c14041cu, 0x0c140c04u, 0x0c140c14u, 0x0c14140cu, 0x0c141c04u,
    0x0c143e14u, 0x0c1c0404u, 0x0c1c0414u, 0x0c1c1404u, 0x0c1c1c0cu, 0x0c1c2434u, 0x0c1c3434u, 0x0c24040cu,
    0x0c24042cu, 0x0c242c04u, 0x0c2c1404u, 0x0c2c1424u, 0x0c2c2434u, 0x0c2c3e0cu, 0x0c34042cu, 0x0c3e1414u,
    0x0c3e2404u, 0x14040404u, 0x14040414u, 0x14040c0cu, 0x14040c1cu, 0x14041404u, 0x14041414u, 0x14041434u,
    0x14041c0cu, 0x14042414u, 0x140c040cu, 0x140c041cu, 0x140c042cu, 0x140c0c04u, 0x140c0c14u, 0x140c140cu,
    0x140c1c04u, 0x140c341cu, 0x140c343eu, 0x140c3e04u, 0x14140404u, 0x14140414u, 0x14140c0cu, 0x14140c3eu,
    0x14141404u, 0x14141414u, 0x14141c3eu, 0x14142404u, 0x14142c2cu, 0x141c040cu, 0x141c0c04u, 0x141c0c24u,
    0x141c3e04u, 0x141c3e24u, 0x14241c2cu, 0x14242c1cu, 0x142c041cu, 0x142c143eu, 0x142c240cu, 0x142c3e24u,
    0x143e040cu, 0x143e041cu, 0x143e0c34u, 0x143e242cu, 0x1c04040cu, 0x1c040c04u, 0x1c040c14u, 0x1c04140cu,
    0x1c04141cu, 0x1c042c04u, 0x1c04342cu, 0x1c043e14u, 0x1c0c0404u, 0x1c0c0414u, 0x1c0c1404u, 0x1c0c1c0cu,
    0x1c0c2424u, 0x1c0c2434u, 0x1c14040cu, 0x1c14041cu, 0x1c140c04u, 0x1c14142cu, 0x1c142c14u, 0x1c143e14u,
    0x1c1c0c0cu, 0x1c1c1c1cu, 0x1c241c04u, 0x1c24243eu, 0x1c243e14u, 0x1c2c0404u, 0x1c2c0434u, 0x1c2c1414u,
    0x1c2c2c2cu, 0x1c340c24u, 0x1c341c34u, 0x1c34341cu, 0x1c3e1c1cu, 0x1c3e3404u, 0x24040424u, 0x24040c3eu,
    0x24041c2cu, 0x24041c3eu, 0x24042c1cu, 0x24042c3eu, 0x240c3e24u, 0x24141404u, 0x24141c3eu, 0x24142404u,
    0x24143404u, 0x24143434u, 0x241c043eu, 0x241c242cu, 0x24240424u, 0x24242c0cu, 0x24243424u, 0x242c142cu,
    0x242c241cu, 0x242c3e04u, 0x243e042cu, 0x243e0c04u, 0x243e0c14u, 0x243e1c04u, 0x2c040c14u, 0x2c04240cu,
    0x2c043e04u, 0x2c0c0404u, 0x2c0c0434u, 0x2c0c1434u, 0x2c0c2c2cu, 0x2c140c24u, 0x2c141c14u, 0x2c143e14u,
    0x2c1c0414u, 0x2c1c2c1cu, 0x2c240c04u, 0x2c24141cu, 0x2c24143eu, 0x2c243e14u, 0x2c2c0414u, 0x2c2c1c0cu,
    0x2c342c04u, 0x2c3e1424u, 0x2c3e2414u, 0x34041424u, 0x34042424u, 0x34042434u, 0x34043424u, 0x340c140cu,
    0x340c340cu, 0x34140c3eu, 0x34143424u, 0x341c1c04u, 0x341c1c34u, 0x34242424u, 0x342c042cu, 0x342c2c14u,
    0x34341c1cu, 0x343e041cu, 0x343e140cu, 0x3e04041cu, 0x3e04042cu, 0x3e04043eu, 0x3e040c04u, 0x3e041c14u,
    0x3e042c14u, 0x3e0c1434u, 0x3e0c2404u, 0x3e140c14u, 0x3e14242cu, 0x3e142c14u, 0x3e1c0404u, 0x3e1c0c2cu,
    0x3e1c1c1cu, 0x3e1c3404u, 0x3e24140cu, 0x3e24240cu, 0x3e2c0404u, 0x3e2c0414u, 0x3e2c1424u, 0x3e341c04u,
};
__device__ __constant__ uint32_t IQ3S_GRID_D[512] = {
    0x01010101u, 0x01010103u, 0x01010105u, 0x0101010bu, 0x0101010fu, 0x01010301u, 0x01010303u, 0x01010305u,
    0x01010309u, 0x0101030du, 0x01010501u, 0x01010503u, 0x0101050bu, 0x01010707u, 0x01010901u, 0x01010905u,
    0x0101090bu, 0x0101090fu, 0x01010b03u, 0x01010b07u, 0x01010d01u, 0x01010d05u, 0x01010f03u, 0x01010f09u,
    0x01010f0fu, 0x01030101u, 0x01030103u, 0x01030105u, 0x01030109u, 0x01030301u, 0x01030303u, 0x0103030bu,
    0x01030501u, 0x01030507u, 0x0103050fu, 0x01030703u, 0x0103070bu, 0x01030909u, 0x01030d03u, 0x01030d0bu,
    0x01030f05u, 0x01050101u, 0x01050103u, 0x0105010bu, 0x0105010fu, 0x01050301u, 0x01050307u, 0x0105030du,
    0x01050503u, 0x0105050bu, 0x01050701u, 0x01050709u, 0x01050905u, 0x0105090bu, 0x0105090fu, 0x01050b03u,
    0x01050b07u, 0x01050f01u, 0x01050f07u, 0x01070107u, 0x01070303u, 0x0107030bu, 0x01070501u, 0x01070505u,
    0x01070703u, 0x01070707u, 0x0107070du, 0x01070909u, 0x01070b01u, 0x01070b05u, 0x01070d0fu, 0x01070f03u,
    0x01070f0bu, 0x01090101u, 0x01090307u, 0x0109030fu, 0x01090503u, 0x01090509u, 0x01090705u, 0x01090901u,
    0x01090907u, 0x01090b03u, 0x01090f01u, 0x010b0105u, 0x010b0109u, 0x010b0501u, 0x010b0505u, 0x010b050du,
    0x010b0707u, 0x010b0903u, 0x010b090bu, 0x010b090fu, 0x010b0d0du, 0x010b0f07u, 0x010d010du, 0x010d0303u,
    0x010d0307u, 0x010d0703u, 0x010d0b05u, 0x010d0f03u, 0x010f0101u, 0x010f0105u, 0x010f0109u, 0x010f0501u,
    0x010f0505u, 0x010f050du, 0x010f0707u, 0x010f0b01u, 0x010f0b09u, 0x03010101u, 0x03010103u, 0x03010105u,
    0x03010109u, 0x03010301u, 0x03010303u, 0x03010307u, 0x0301030bu, 0x0301030fu, 0x03010501u, 0x03010505u,
    0x03010703u, 0x03010709u, 0x0301070du, 0x03010b09u, 0x03010b0du, 0x03010d03u, 0x03010f05u, 0x03030101u,
    0x03030103u, 0x03030107u, 0x0303010du, 0x03030301u, 0x03030309u, 0x03030503u, 0x03030701u, 0x03030707u,
    0x03030903u, 0x03030b01u, 0x03030b05u, 0x03030f01u, 0x03030f0du, 0x03050101u, 0x03050305u, 0x0305030bu,
    0x0305030fu, 0x03050501u, 0x03050509u, 0x03050705u, 0x03050901u, 0x03050907u, 0x03050b0bu, 0x03050d01u,
    0x03050f05u, 0x03070103u, 0x03070109u, 0x0307010fu, 0x03070301u, 0x03070307u, 0x03070503u, 0x0307050fu,
    0x03070701u, 0x03070709u, 0x03070903u, 0x03070d05u, 0x03070f01u, 0x03090107u, 0x0309010bu, 0x03090305u,
    0x03090309u, 0x03090703u, 0x03090707u, 0x03090905u, 0x0309090du, 0x03090b01u, 0x03090b09u, 0x030b0103u,
    0x030b0301u, 0x030b0307u, 0x030b0503u, 0x030b0701u, 0x030b0705u, 0x030b0b03u, 0x030d0501u, 0x030d0509u,
    0x030d050fu, 0x030d0909u, 0x030d090du, 0x030f0103u, 0x030f0107u, 0x030f0301u, 0x030f0305u, 0x030f0503u,
    0x030f070bu, 0x030f0903u, 0x030f0d05u, 0x030f0f01u, 0x05010101u, 0x05010103u, 0x05010107u, 0x0501010bu,
    0x0501010fu, 0x05010301u, 0x05010305u, 0x05010309u, 0x0501030du, 0x05010503u, 0x05010507u, 0x0501050fu,
    0x05010701u, 0x05010705u, 0x05010903u, 0x05010907u, 0x0501090bu, 0x05010b01u, 0x05010b05u, 0x05010d0fu,
    0x05010f01u, 0x05010f07u, 0x05010f0bu, 0x05030101u, 0x05030105u, 0x05030301u, 0x05030307u, 0x0503030fu,
    0x05030505u, 0x0503050bu, 0x05030703u, 0x05030709u, 0x05030905u, 0x05030b03u, 0x05050103u, 0x05050109u,
    0x0505010fu, 0x05050503u, 0x05050507u, 0x05050701u, 0x0505070fu, 0x05050903u, 0x05050b07u, 0x05050b0fu,
    0x05050f03u, 0x05050f09u, 0x05070101u, 0x05070105u, 0x0507010bu, 0x05070303u, 0x05070505u, 0x05070509u,
    0x05070703u, 0x05070707u, 0x05070905u, 0x05070b01u, 0x05070d0du, 0x05090103u, 0x0509010fu, 0x05090501u,
    0x05090507u, 0x05090705u, 0x0509070bu, 0x05090903u, 0x05090f05u, 0x05090f0bu, 0x050b0109u, 0x050b0303u,
    0x050b0505u, 0x050b070fu, 0x050b0901u, 0x050b0b07u, 0x050b0f01u, 0x050d0101u, 0x050d0105u, 0x050d010fu,
    0x050d0503u, 0x050d0b0bu, 0x050d0d03u, 0x050f010bu, 0x050f0303u, 0x050f050du, 0x050f0701u, 0x050f0907u,
    0x050f0b01u, 0x07010105u, 0x07010303u, 0x07010307u, 0x0701030bu, 0x0701030fu, 0x07010505u, 0x07010703u,
    0x07010707u, 0x0701070bu, 0x07010905u, 0x07010909u, 0x0701090fu, 0x07010b03u, 0x07010d07u, 0x07010f03u,
    0x07030103u, 0x07030107u, 0x0703010bu, 0x07030309u, 0x07030503u, 0x07030507u, 0x07030901u, 0x07030d01u,
    0x07030f05u, 0x07030f0du, 0x07050101u, 0x07050305u, 0x07050501u, 0x07050705u, 0x07050709u, 0x07050b01u,
    0x07070103u, 0x07070301u, 0x07070309u, 0x07070503u, 0x07070507u, 0x0707050fu, 0x07070701u, 0x07070903u,
    0x07070907u, 0x0707090fu, 0x07070b0bu, 0x07070f07u, 0x07090107u, 0x07090303u, 0x0709030du, 0x07090505u,
    0x07090703u, 0x07090b05u, 0x07090d01u, 0x07090d09u, 0x070b0103u, 0x070b0301u, 0x070b0305u, 0x070b050bu,
    0x070b0705u, 0x070b0909u, 0x070b0b0du, 0x070b0f07u, 0x070d030du, 0x070d0903u, 0x070f0103u, 0x070f0107u,
    0x070f0501u, 0x070f0505u, 0x070f070bu, 0x09010101u, 0x09010109u, 0x09010305u, 0x09010501u, 0x09010509u,
    0x0901050fu, 0x09010705u, 0x09010903u, 0x09010b01u, 0x09010f01u, 0x09030105u, 0x0903010fu, 0x09030303u,
    0x09030307u, 0x09030505u, 0x09030701u, 0x0903070bu, 0x09030907u, 0x09030b03u, 0x09030b0bu, 0x09050103u,
    0x09050107u, 0x09050301u, 0x0905030bu, 0x09050503u, 0x09050707u, 0x09050901u, 0x09050b0fu, 0x09050d05u,
    0x09050f01u, 0x09070109u, 0x09070303u, 0x09070307u, 0x09070501u, 0x09070505u, 0x09070703u, 0x0907070bu,
    0x09090101u, 0x09090105u, 0x09090509u, 0x0909070fu, 0x09090901u, 0x09090f03u, 0x090b010bu, 0x090b010fu,
    0x090b0503u, 0x090b0d05u, 0x090d0307u, 0x090d0709u, 0x090d0d01u, 0x090f0301u, 0x090f030bu, 0x090f0701u,
    0x090f0907u, 0x090f0b03u, 0x0b010105u, 0x0b010301u, 0x0b010309u, 0x0b010505u, 0x0b010901u, 0x0b010909u,
    0x0b01090fu, 0x0b010b05u, 0x0b010d0du, 0x0b010f09u, 0x0b030103u, 0x0b030107u, 0x0b03010bu, 0x0b030305u,
    0x0b030503u, 0x0b030705u, 0x0b030f05u, 0x0b050101u, 0x0b050303u, 0x0b050507u, 0x0b050701u, 0x0b05070du,
    0x0b050b07u, 0x0b070105u, 0x0b07010fu, 0x0b070301u, 0x0b07050fu, 0x0b070909u, 0x0b070b03u, 0x0b070d0bu,
    0x0b070f07u, 0x0b090103u, 0x0b090109u, 0x0b090501u, 0x0b090705u, 0x0b09090du, 0x0b0b0305u, 0x0b0b050du,
    0x0b0b0b03u, 0x0b0b0b07u, 0x0b0d0905u, 0x0b0f0105u, 0x0b0f0109u, 0x0b0f0505u, 0x0d010303u, 0x0d010307u,
    0x0d01030bu, 0x0d010703u, 0x0d010707u, 0x0d010d01u, 0x0d030101u, 0x0d030501u, 0x0d03050fu, 0x0d030d09u,
    0x0d050305u, 0x0d050709u, 0x0d050905u, 0x0d050b0bu, 0x0d050d05u, 0x0d050f01u, 0x0d070101u, 0x0d070309u,
    0x0d070503u, 0x0d070901u, 0x0d09050bu, 0x0d090907u, 0x0d090d05u, 0x0d0b0101u, 0x0d0b0107u, 0x0d0b0709u,
    0x0d0b0d01u, 0x0d0d010bu, 0x0d0d0901u, 0x0d0f0303u, 0x0d0f0307u, 0x0f010101u, 0x0f010109u, 0x0f01010fu,
    0x0f010501u, 0x0f010505u, 0x0f01070du, 0x0f010901u, 0x0f010b09u, 0x0f010d05u, 0x0f030105u, 0x0f030303u,
    0x0f030509u, 0x0f030907u, 0x0f03090bu, 0x0f050103u, 0x0f050109u, 0x0f050301u, 0x0f05030du, 0x0f050503u,
    0x0f050701u, 0x0f050b03u, 0x0f070105u, 0x0f070705u, 0x0f07070bu, 0x0f070b07u, 0x0f090103u, 0x0f09010bu,
    0x0f090307u, 0x0f090501u, 0x0f090b01u, 0x0f0b0505u, 0x0f0b0905u, 0x0f0d0105u, 0x0f0d0703u, 0x0f0f0101u,
};
__device__ __constant__ uint8_t KSIGNS_IQ2XS_D[128] = {
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15,
    144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159,
    160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175,
    48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207,
    80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
    96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};
#endif // __HIPCC__

// Decode weight type ids + the qstore store-dispatch are defined ONCE in the preamble near the top
// (shared by the dp4a core); the per-type qdw_traits below carry the scalar core's BYTES/ELEMS/
// SYMMETRIC/NSC. One row per wired type.
template <int WTYPE> struct qdw_traits;
template <> struct qdw_traits<DW_Q8_0>   { static constexpr int BYTES = 34;  static constexpr int ELEMS = 32;  static constexpr bool SYMMETRIC = true;  static constexpr int NSC = 1; };
template <> struct qdw_traits<DW_Q4_0>   { static constexpr int BYTES = 18;  static constexpr int ELEMS = 32;  static constexpr bool SYMMETRIC = true;  static constexpr int NSC = 1; };
template <> struct qdw_traits<DW_Q4_K>   { static constexpr int BYTES = 144; static constexpr int ELEMS = 256; static constexpr bool SYMMETRIC = false; static constexpr int NSC = 8; };
template <> struct qdw_traits<DW_Q6_K>   { static constexpr int BYTES = 210; static constexpr int ELEMS = 256; static constexpr bool SYMMETRIC = true;  static constexpr int NSC = 16; };
template <> struct qdw_traits<DW_IQ4_XS> { static constexpr int BYTES = 136; static constexpr int ELEMS = 256; static constexpr bool SYMMETRIC = true;  static constexpr int NSC = 8; };
template <> struct qdw_traits<DW_TQ2_0>  { static constexpr int BYTES = 66;  static constexpr int ELEMS = 256; static constexpr bool SYMMETRIC = true;  static constexpr int NSC = 1; };
template <> struct qdw_traits<DW_Q2_K>   { static constexpr int BYTES = 84;  static constexpr int ELEMS = 256; static constexpr bool SYMMETRIC = false; static constexpr int NSC = 16; };
template <> struct qdw_traits<DW_Q3_K>   { static constexpr int BYTES = 110; static constexpr int ELEMS = 256; static constexpr bool SYMMETRIC = true;  static constexpr int NSC = 16; };
template <> struct qdw_traits<DW_Q5_K>   { static constexpr int BYTES = 176; static constexpr int ELEMS = 256; static constexpr bool SYMMETRIC = false; static constexpr int NSC = 8; };
template <> struct qdw_traits<DW_Q4_1>   { static constexpr int BYTES = 20;  static constexpr int ELEMS = 32;  static constexpr bool SYMMETRIC = false; static constexpr int NSC = 1; };
template <> struct qdw_traits<DW_Q5_0>   { static constexpr int BYTES = 22;  static constexpr int ELEMS = 32;  static constexpr bool SYMMETRIC = true;  static constexpr int NSC = 1; };
template <> struct qdw_traits<DW_Q5_1>   { static constexpr int BYTES = 24;  static constexpr int ELEMS = 32;  static constexpr bool SYMMETRIC = false; static constexpr int NSC = 1; };
template <> struct qdw_traits<DW_Q8_1>   { static constexpr int BYTES = 36;  static constexpr int ELEMS = 32;  static constexpr bool SYMMETRIC = true;  static constexpr int NSC = 1; };
template <> struct qdw_traits<DW_IQ2_XXS>{ static constexpr int BYTES = 66;  static constexpr int ELEMS = 256; static constexpr bool SYMMETRIC = true;  static constexpr int NSC = 8; };
template <> struct qdw_traits<DW_IQ2_XS> { static constexpr int BYTES = 74;  static constexpr int ELEMS = 256; static constexpr bool SYMMETRIC = true;  static constexpr int NSC = 8; };
template <> struct qdw_traits<DW_IQ2_S>  { static constexpr int BYTES = 82;  static constexpr int ELEMS = 256; static constexpr bool SYMMETRIC = true;  static constexpr int NSC = 8; };
template <> struct qdw_traits<DW_IQ3_XXS>{ static constexpr int BYTES = 98;  static constexpr int ELEMS = 256; static constexpr bool SYMMETRIC = true;  static constexpr int NSC = 8; };
template <> struct qdw_traits<DW_IQ3_S>  { static constexpr int BYTES = 110; static constexpr int ELEMS = 256; static constexpr bool SYMMETRIC = true;  static constexpr int NSC = 8; };
template <> struct qdw_traits<DW_IQ4_NL> { static constexpr int BYTES = 18;  static constexpr int ELEMS = 32;  static constexpr bool SYMMETRIC = true;  static constexpr int NSC = 1; };
template <> struct qdw_traits<DW_TQ1_0>  { static constexpr int BYTES = 54;  static constexpr int ELEMS = 256; static constexpr bool SYMMETRIC = true;  static constexpr int NSC = 1; };
template <> struct qdw_traits<DW_IQ1_S>  { static constexpr int BYTES = 50;  static constexpr int ELEMS = 256; static constexpr bool SYMMETRIC = false; static constexpr int NSC = 8; };
template <> struct qdw_traits<DW_IQ1_M>  { static constexpr int BYTES = 56;  static constexpr int ELEMS = 256; static constexpr bool SYMMETRIC = false; static constexpr int NSC = 8; };
template <> struct qdw_traits<DW_MXFP4>  { static constexpr int BYTES = 17;  static constexpr int ELEMS = 32;  static constexpr bool SYMMETRIC = true;  static constexpr int NSC = 1; };

// Per-type decode. `partial<XT>(blk, xb, lane)` returns lane L's partial dot over its owned
// positions { e*32 + L } of ONE block, mirroring the CPU `to_float` for that type bit-for-bit.
// Function templates cannot partial-specialize, so each type is a struct with a templated method.
template <int WTYPE> struct qdec;

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

// Q2_K: ASYMMETRIC 2-bit. Block = scales[16], qs[64], d (f16), dmin (f16) = 84 B. Lane L owns
// positions { e*32 + L }, e in 0..8 = 32-block J=e. scale index is = 2*e + (L>=16); dl = d*(sc&0xF),
// ml = dmin*(sc>>4); q = (qs[(e>>2)*32 + L] >> (2*(e&3))) & 3; val = dl*q - ml. (BlockQ2K::to_float.)
template <> struct qdec<DW_Q2_K> {
    template <typename XT>
    static __device__ __forceinline__ float partial(const uint8_t* __restrict__ blk, const XT* __restrict__ xb, int lane) {
        const uint8_t* scales = blk;          // 16
        const uint8_t* qs     = blk + 16;     // 64
        const float d    = __half2float(*reinterpret_cast<const __half*>(blk + 80));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(blk + 82));
        const int half = lane >> 4;           // 0/1 -> low/high 16 of the 32-block
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int shift = 2 * (e & 3);
            const uint8_t sc = scales[e * 2 + half];
            const float dl = d    * (float)(sc & 0xF);
            const float ml = dmin * (float)(sc >> 4);
            const int q = (qs[(e >> 2) * 32 + lane] >> shift) & 3;
            s += (dl * (float)q - ml) * (float)xb[e * 32 + lane];
        }
        return s;
    }
};

// Q3_K: symmetric 3-bit, signed per-16 scale. Block = hmask[32], qs[64], scales[12], d (f16) = 110 B.
// Lane L owns { e*32 + L }, e = 32-block J. scale index is = 2*e + (L>=16); 3-bit q = (low2 | hbit<<2)
// - 4, low2 = (qs[(e>>2)*32 + L] >> (2*(e&3)))&3, hbit = (hmask[L]>>e)&1; val = d*(sc[is]-32)*q.
template <> struct qdec<DW_Q3_K> {
    template <typename XT>
    static __device__ __forceinline__ float partial(const uint8_t* __restrict__ blk, const XT* __restrict__ xb, int lane) {
        const uint8_t* hmask = blk;           // 32
        const uint8_t* qs    = blk + 32;      // 64
        int8_t sc[16];
        q3k_scales(blk + 96, sc);             // 12 scale bytes -> 16 i8
        const float d = __half2float(*reinterpret_cast<const __half*>(blk + 108));
        const int half = lane >> 4;
        const uint8_t hm = hmask[lane];       // hmask byte index = position within 32-block = lane
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int shift = 2 * (e & 3);
            const float dl = d * (float)((int)sc[e * 2 + half] - 32);
            const int low2 = (qs[(e >> 2) * 32 + lane] >> shift) & 3;
            const int hbit = (hm >> e) & 1;
            const int q = (low2 | (hbit << 2)) - 4;
            s += dl * (float)q * (float)xb[e * 32 + lane];
        }
        return s;
    }
};

// Q5_K: ASYMMETRIC 5-bit super-block. block = {d, dmin, 12 scale bytes, qh[32], qs[128]} = 176 B.
// Mirrors Q4_K (val = d*sc*q - dmin*m) but q is 5-bit: for sub-block e (0..8), position l=lane, the
// qs byte is qs[(e>>1)*32 + l] (low nibble for even e, high for odd e), the 5th bit is qh[l] bit e,
// and (sc,m) = get_scale_min_k4(e). Lane L owns positions { e*32 + L }. (BlockQ5K::to_float.)
template <> struct qdec<DW_Q5_K> {
    template <typename XT>
    static __device__ __forceinline__ float partial(const uint8_t* __restrict__ blk, const XT* __restrict__ xb, int lane) {
        const float d    = __half2float(*reinterpret_cast<const __half*>(blk));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(blk + 2));
        const uint8_t* scales = blk + 4;       // 12 packed scale bytes (same layout as Q4_K)
        const uint8_t* qh = blk + 16;          // 32 high-bit bytes
        const uint8_t* qs = blk + 48;          // 128 quant bytes
        const uint8_t hbit = qh[lane];
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            int sc, m;
            q4k_scale_min(scales, e, &sc, &m);
            const uint8_t qb = qs[(e >> 1) * 32 + lane];
            const int nib = (e & 1) ? (qb >> 4) : (qb & 0xF);
            const int hi  = ((hbit >> e) & 1) << 4;
            const float val = d * (float)sc * (float)(nib | hi) - dmin * (float)m;
            s += val * (float)xb[e * 32 + lane];
        }
        return s;
    }
};

// Q4_1: ASYMMETRIC legacy. d (f16) at 0, m (f16) at 2, 16 nibble-pairs at +4. pos p (=lane): p<16 ->
// low nibble qs[p], p>=16 -> high nibble qs[p-16]; val = nib*d + m. (BlockQ4_1::to_float.)
template <> struct qdec<DW_Q4_1> {
    template <typename XT>
    static __device__ __forceinline__ float partial(const uint8_t* __restrict__ blk, const XT* __restrict__ xb, int lane) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const float m = __half2float(*reinterpret_cast<const __half*>(blk + 2));
        const uint8_t* qs = blk + 4;
        const int nib = (lane < 16) ? (qs[lane] & 0x0F) : (qs[lane - 16] >> 4);
        return ((float)nib * d + m) * (float)xb[lane];
    }
};

// Q5_0: symmetric legacy 5-bit. d (f16) at 0, qh[4] (u32 LE) at +2, 16 nibble-pairs at +6. pos p
// (=lane): nib = p<16 ? qs[p]&0xF : qs[p-16]>>4; the 5th bit is qh-bit p (low p<16 and high p>=16
// BOTH index bit p -- the to_float xh_0=bit j / xh_1=bit j+16 walk); val = ((nib|bit<<4)-16)*d. qh
// is read byte-wise (the 22 B stride leaves blk+2 only 2-aligned, so a u32 load could misalign).
template <> struct qdec<DW_Q5_0> {
    template <typename XT>
    static __device__ __forceinline__ float partial(const uint8_t* __restrict__ blk, const XT* __restrict__ xb, int lane) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint8_t* qhb = blk + 2;
        const uint32_t qh = (uint32_t)qhb[0] | ((uint32_t)qhb[1] << 8) | ((uint32_t)qhb[2] << 16) | ((uint32_t)qhb[3] << 24);
        const uint8_t* qs = blk + 6;
        const int nib = (lane < 16) ? (qs[lane] & 0x0F) : (qs[lane - 16] >> 4);
        const int hi  = (int)((qh >> lane) & 1u) << 4;
        return (float)((nib | hi) - 16) * d * (float)xb[lane];
    }
};

// Q5_1: ASYMMETRIC legacy 5-bit. d (f16) at 0, m (f16) at 2, qh[4] (u32 LE) at +4, 16 nibble-pairs
// at +8. Like Q5_0 (5th bit = qh-bit p) but NOT centered and with a min: val = (nib|bit<<4)*d + m.
template <> struct qdec<DW_Q5_1> {
    template <typename XT>
    static __device__ __forceinline__ float partial(const uint8_t* __restrict__ blk, const XT* __restrict__ xb, int lane) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const float m = __half2float(*reinterpret_cast<const __half*>(blk + 2));
        const uint8_t* qhb = blk + 4;
        const uint32_t qh = (uint32_t)qhb[0] | ((uint32_t)qhb[1] << 8) | ((uint32_t)qhb[2] << 16) | ((uint32_t)qhb[3] << 24);
        const uint8_t* qs = blk + 8;
        const int nib = (lane < 16) ? (qs[lane] & 0x0F) : (qs[lane - 16] >> 4);
        const int hi  = (int)((qh >> lane) & 1u) << 4;
        return ((float)(nib | hi) * d + m) * (float)xb[lane];
    }
};

// Q8_1: symmetric 8-bit (Q8_0 + a per-block sum field). d (f16) at 0, s (f16, the sum -- unused for
// dequant) at +2, 32 int8 quants at +4. pos p (=lane): val = d * qs[p]. (BlockQ8_1::to_float.)
template <> struct qdec<DW_Q8_1> {
    template <typename XT>
    static __device__ __forceinline__ float partial(const uint8_t* __restrict__ blk, const XT* __restrict__ xb, int lane) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 4);
        return d * (float)qs[lane] * (float)xb[lane];
    }
};

// IQ-quant codebook decode (BlockIQ2xxs/xs/s, BlockIQ3xxs/s). Each 256-elem super-block has 8 sub-
// blocks of 32 (ib32=e). Within a 32, the CPU to_float writes 4 groups of 8 (l in 0..4), so lane L
// (its owned position e*32+L) maps to group l=L>>3 and within-group index L&7. The grid byte (unsigned
// magnitude 0..255) times the per-sub-block scale `db` times the +-1 sign IS the dequantized weight --
// no rounding, integer-exact to the CPU oracle. Grids/ksigns are the __constant__ tables generated
// from iq_grids.rs above. (sign bit set -> negative, matching kmask(j) in iq_quants.rs.)

// IQ2_XXS: d@0, qs[32]u16@2. Per e: 4 u16 -> aux32_0 (4 grid-idx bytes) + aux32_1 (4 7-bit ksign
// indices + 4-bit scale in top nibble). db = d*(0.5 + (aux32_1>>28))*0.25. (BlockIQ2xxs::to_float)
template <> struct qdec<DW_IQ2_XXS> {
    template <typename XT>
    static __device__ __forceinline__ float partial(const uint8_t* __restrict__ blk, const XT* __restrict__ xb, int lane) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint16_t* qs = reinterpret_cast<const uint16_t*>(blk + 2);
        const int l = lane >> 3;              // group 0..3
        const int j = lane & 7;               // element-in-group 0..7
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const uint32_t aux32_0 = (uint32_t)qs[4 * e] | ((uint32_t)qs[4 * e + 1] << 16);
            const uint32_t aux32_1 = (uint32_t)qs[4 * e + 2] | ((uint32_t)qs[4 * e + 3] << 16);
            const int gi = (int)((aux32_0 >> (8 * l)) & 0xFF);
            const float db = d * (0.5f + (float)(aux32_1 >> 28)) * 0.25f;
            const uint64_t entry = IQ2XXS_GRID_D[gi];
            const uint8_t signs = KSIGNS_IQ2XS_D[(aux32_1 >> (7 * l)) & 127];
            const float g = (float)((entry >> (8 * j)) & 0xFF);
            const float sign = (signs & (1u << j)) ? -1.0f : 1.0f;
            s += db * g * sign * (float)xb[e * 32 + lane];
        }
        return s;
    }
};

// IQ2_XS: d@0, qs[32]u16@2, scales[8]u8@66. idx=qv&511, ksign=qv>>9; db = d*(0.5 + scale_nibble)*0.25,
// the nibble = sc&0xf for groups 0,1 and sc>>4 for groups 2,3. (BlockIQ2xs::to_float)
template <> struct qdec<DW_IQ2_XS> {
    template <typename XT>
    static __device__ __forceinline__ float partial(const uint8_t* __restrict__ blk, const XT* __restrict__ xb, int lane) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint16_t* qs = reinterpret_cast<const uint16_t*>(blk + 2);
        const uint8_t* scales = blk + 66;
        const int l = lane >> 3;
        const int j = lane & 7;
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const uint8_t sc = scales[e];
            const int nib = (l < 2) ? (sc & 0xF) : (sc >> 4);
            const float db = d * (0.5f + (float)nib) * 0.25f;
            const uint16_t qv = qs[4 * e + l];
            const uint64_t entry = IQ2XS_GRID_D[qv & 511];
            const uint8_t signs = KSIGNS_IQ2XS_D[qv >> 9];
            const float g = (float)((entry >> (8 * j)) & 0xFF);
            const float sign = (signs & (1u << j)) ? -1.0f : 1.0f;
            s += db * g * sign * (float)xb[e * 32 + lane];
        }
        return s;
    }
};

// IQ2_S: d@0, qs[64]u8@2 (low 32 = grid idx low byte, high 32 = sign bytes), qh[8]@66, scales[8]@74.
// idx = qs[4e+l] | ((qh[e] << (8-2l)) & 0x300) (10-bit); signs from qs[32+4e+l]. (BlockIQ2s::to_float)
template <> struct qdec<DW_IQ2_S> {
    template <typename XT>
    static __device__ __forceinline__ float partial(const uint8_t* __restrict__ blk, const XT* __restrict__ xb, int lane) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint8_t* qs = blk + 2;          // [0..32) idx low byte, [32..64) sign bytes
        const uint8_t* qh = blk + 66;
        const uint8_t* scales = blk + 74;
        const int l = lane >> 3;
        const int j = lane & 7;
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const uint8_t sc = scales[e];
            const int nib = (l < 2) ? (sc & 0xF) : (sc >> 4);
            const float db = d * (0.5f + (float)nib) * 0.25f;
            const int idx = (int)qs[4 * e + l] | (((int)qh[e] << (8 - 2 * l)) & 0x300);
            const uint64_t entry = IQ2S_GRID_D[idx];
            const uint8_t signs = qs[32 + 4 * e + l];
            const float g = (float)((entry >> (8 * j)) & 0xFF);
            const float sign = (signs & (1u << j)) ? -1.0f : 1.0f;
            s += db * g * sign * (float)xb[e * 32 + lane];
        }
        return s;
    }
};

// IQ3_XXS: d@0, qs[96]u8@2 = grid indices [0..64) + scales_and_signs [64..96) (u32 LE per e). u32
// grid (4 bytes), db = d*(0.5 + (aux32>>28))*0.5. Each group l: 2 grid bytes -> 4 elems each. The 8
// within-group positions split m<4 -> grid byte qs[8e+2l], m>=4 -> qs[8e+2l+1]. (BlockIQ3xxs::to_float)
template <> struct qdec<DW_IQ3_XXS> {
    template <typename XT>
    static __device__ __forceinline__ float partial(const uint8_t* __restrict__ blk, const XT* __restrict__ xb, int lane) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint8_t* qs = blk + 2;          // grid indices [0..64)
        const uint8_t* ss = blk + 2 + 64;     // scales_and_signs [0..32)
        const int l = lane >> 3;
        const int m = lane & 7;
        const int half = (m < 4) ? 0 : 1;     // qs[8e+2l] vs qs[8e+2l+1]
        const int jj = m & 3;                 // byte within the u32 grid entry
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const uint32_t aux32 = (uint32_t)ss[4 * e] | ((uint32_t)ss[4 * e + 1] << 8)
                                 | ((uint32_t)ss[4 * e + 2] << 16) | ((uint32_t)ss[4 * e + 3] << 24);
            const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
            const uint8_t signs = KSIGNS_IQ2XS_D[(aux32 >> (7 * l)) & 127];
            const uint32_t entry = IQ3XXS_GRID_D[qs[8 * e + 2 * l + half]];
            const float g = (float)((entry >> (8 * jj)) & 0xFF);
            const float sign = (signs & (1u << m)) ? -1.0f : 1.0f;
            s += db * g * sign * (float)xb[e * 32 + lane];
        }
        return s;
    }
};

// IQ3_S: d@0, qs[64]u8@2, qh[8]@66, signs[32]@74, scales[4]@106. u32 grid, db = d*(1 + 2*scale_nibble)
// (nibble = sc&0xf for even sub-blocks, sc>>4 for odd; sc = scales[e>>1]). idx bit-8 comes from qh:
// m<4 uses (8-2l), m>=4 uses (7-2l). signs from signs[4e+l]. (BlockIQ3s::to_float)
template <> struct qdec<DW_IQ3_S> {
    template <typename XT>
    static __device__ __forceinline__ float partial(const uint8_t* __restrict__ blk, const XT* __restrict__ xb, int lane) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint8_t* qs = blk + 2;
        const uint8_t* qh = blk + 66;
        const uint8_t* signs_b = blk + 74;
        const uint8_t* scales = blk + 106;
        const int l = lane >> 3;
        const int m = lane & 7;
        const int half = (m < 4) ? 0 : 1;
        const int jj = m & 3;
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const uint8_t sc = scales[e >> 1];
            const int nib = (e & 1) ? (sc >> 4) : (sc & 0xF);
            const float db = d * (1.0f + 2.0f * (float)nib);
            const int shift = (m < 4) ? (8 - 2 * l) : (7 - 2 * l);
            const int idx = (int)qs[8 * e + 2 * l + half] | (((int)qh[e] << shift) & 256);
            const uint32_t entry = IQ3S_GRID_D[idx];
            const uint8_t signs = signs_b[4 * e + l];
            const float g = (float)((entry >> (8 * jj)) & 0xFF);
            const float sign = (signs & (1u << m)) ? -1.0f : 1.0f;
            s += db * g * sign * (float)xb[e * 32 + lane];
        }
        return s;
    }
};

// IQ4_NL: symmetric codebook, 32 elems (ELEMS/32 = 1, so position = lane). d (f16) at 0, qs[16] at +2.
// pos p<16 -> low nibble qs[p]; p>=16 -> high nibble qs[p-16]; val = d*KVALUES_IQ4NL[idx]. Same layout
// as Q4_0 with the codebook LUT in place of (nibble-8). (BlockIQ4nl::to_float, k_quants.rs:764.)
template <> struct qdec<DW_IQ4_NL> {
    template <typename XT>
    static __device__ __forceinline__ float partial(const uint8_t* __restrict__ blk, const XT* __restrict__ xb, int lane) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint8_t* qs = blk + 2;
        const int idx = (lane < 16) ? (qs[lane] & 0x0F) : (qs[lane - 16] >> 4);
        return d * (float)KVALUES_IQ4NL_D[idx] * (float)xb[lane];
    }
};

// MXFP4 (gpt-oss): E8M0 1-byte scale at blk[0], qs at blk+1, FP4 codebook. Lane L owns element L:
// low nibbles fill 0..15, high nibbles 16..31 (same split as IQ4_NL). Mirrors dequantize_row_mxfp4.
template <> struct qdec<DW_MXFP4> {
    template <typename XT>
    static __device__ __forceinline__ float partial(const uint8_t* __restrict__ blk, const XT* __restrict__ xb, int lane) {
        const float d = e8m0_to_fp32_half(blk[0]);
        const uint8_t* qs = blk + 1;
        const int idx = (lane < 16) ? (qs[lane] & 0x0F) : (qs[lane - 16] >> 4);
        return d * (float)KVALUES_MXFP4_D[idx] * (float)xb[lane];
    }
};

// TQ1_0: symmetric ternary base-3 (1.69 bpw), 256 elems. qs[48] at 0, qh[4] at +48, d (f16) at +52.
// base-3 unpack: q = (byte * pow3[n]) mod 256; xi = (q*3)>>8 in [0,2]; val = (xi-1)*d. Output pos p maps
// to (byte,n): p<160 main (n=p/32, byte=qs[p&31]); 160<=p<240 tail (p2=p-160, n=p2/16, byte=qs[32+p2%16]);
// p>=240 qh (p3=p-240, n=p3/4, byte=qh[p3%4]). pow3[5]=243 is never used. (BlockTQ1_0::to_float.)
template <> struct qdec<DW_TQ1_0> {
    template <typename XT>
    static __device__ __forceinline__ float partial(const uint8_t* __restrict__ blk, const XT* __restrict__ xb, int lane) {
        const uint8_t* qs = blk;              // 48
        const uint8_t* qh = blk + 48;         // 4
        const float d = __half2float(*reinterpret_cast<const __half*>(blk + 52));
        const uint8_t pow3[5] = {1, 3, 9, 27, 81};
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int p = e * 32 + lane;
            uint8_t b; int n;
            if (p < 160)      { n = p >> 5;             b = qs[p & 31]; }
            else if (p < 240) { const int p2 = p - 160; n = p2 >> 4; b = qs[32 + (p2 & 15)]; }
            else              { const int p3 = p - 240; n = p3 >> 2; b = qh[p3 & 3]; }
            const uint16_t q = (uint16_t)(uint8_t)(b * pow3[n]);
            const int xi = (int)((q * 3) >> 8);
            const float val = (float)(xi - 1) * d;
            s += val * (float)xb[p];
        }
        return s;
    }
};

// IQ1_S: signed-grid + delta bias (1.56 bpw), 256 elems. d (f16) at 0, qs[32] at +2, qh[8] (u16) at +34.
// 8 sub-blocks (ib=e) of 32; within a sub-block 4 groups (l=lane/8) of 8 (j=lane%8). dl = d*(2*((qh>>12)
// &7)+1); delta = (qh&0x8000)? -IQ1S_DELTA : +IQ1S_DELTA; idx = qs[ib*4+l] | (((qh>>3l)&7)<<8); g = signed
// grid byte j; val = dl*(g + delta). (BlockIQ1s::to_float, iq_quants.rs:525.)
template <> struct qdec<DW_IQ1_S> {
    template <typename XT>
    static __device__ __forceinline__ float partial(const uint8_t* __restrict__ blk, const XT* __restrict__ xb, int lane) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint8_t* qs = blk + 2;                                       // 32
        const uint16_t* qh = reinterpret_cast<const uint16_t*>(blk + 34);  // 8 u16
        const int l = lane >> 3;     // group 0..3
        const int j = lane & 7;      // byte 0..7
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const uint16_t qhv = qh[e];
            const float dl = d * (2.0f * (float)((qhv >> 12) & 7) + 1.0f);
            const float delta = (qhv & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;
            const int idx = (int)qs[e * 4 + l] | ((((int)(qhv >> (3 * l))) & 7) << 8);
            const int8_t g = (int8_t)(uint8_t)(IQ1S_GRID_D[idx] >> (8 * j));
            const float val = dl * ((float)g + delta);
            s += val * (float)xb[e * 32 + lane];
        }
        return s;
    }
};

// IQ1_M: signed-grid + delta bias (1.75 bpw), 256 elems, NO top-level d -- the f16 super-scale is
// reconstructed from the top nibbles of scales[8] (as u16[4]). qs[32] at 0, qh[16] at +32, scales[8] at
// +48. Per sub-block ib=e: dl1/dl2 from 3-bit fields of sc[ib/2]; group l=lane/8 picks idx from
// qs[ib*4+l] | (qh-nibble<<8), delta from a qh bit; groups 0,1 use dl1, 2,3 use dl2. (BlockIQ1m::to_float.)
template <> struct qdec<DW_IQ1_M> {
    template <typename XT>
    static __device__ __forceinline__ float partial(const uint8_t* __restrict__ blk, const XT* __restrict__ xb, int lane) {
        const uint8_t* qs = blk;             // 32
        const uint8_t* qh = blk + 32;        // 16
        const uint8_t* scales = blk + 48;    // 8
        const uint16_t sc0 = (uint16_t)scales[0] | ((uint16_t)scales[1] << 8);
        const uint16_t sc1 = (uint16_t)scales[2] | ((uint16_t)scales[3] << 8);
        const uint16_t sc2 = (uint16_t)scales[4] | ((uint16_t)scales[5] << 8);
        const uint16_t sc3 = (uint16_t)scales[6] | ((uint16_t)scales[7] << 8);
        const uint16_t scale_u16 = (uint16_t)((sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000));
        const float d = __half2float(__ushort_as_half(scale_u16));
        const uint16_t sc[4] = {sc0, sc1, sc2, sc3};
        const int l = lane >> 3;     // group 0..3
        const int j = lane & 7;      // byte 0..7
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const uint16_t scv = sc[e >> 1];
            const float dl1 = d * (2.0f * (float)((scv >> (6 * (e & 1)))     & 0x7) + 1.0f);
            const float dl2 = d * (2.0f * (float)((scv >> (6 * (e & 1) + 3)) & 0x7) + 1.0f);
            const uint8_t qh0 = qh[e * 2];
            const uint8_t qh1 = qh[e * 2 + 1];
            int idx; float delta; float dl;
            if (l == 0)      { idx = (int)qs[e * 4]     | (((int)qh0 << 8) & 0x700); delta = (qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA; dl = dl1; }
            else if (l == 1) { idx = (int)qs[e * 4 + 1] | (((int)qh0 << 4) & 0x700); delta = (qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA; dl = dl1; }
            else if (l == 2) { idx = (int)qs[e * 4 + 2] | (((int)qh1 << 8) & 0x700); delta = (qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA; dl = dl2; }
            else             { idx = (int)qs[e * 4 + 3] | (((int)qh1 << 4) & 0x700); delta = (qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA; dl = dl2; }
            const int8_t g = (int8_t)(uint8_t)(IQ1S_GRID_D[idx] >> (8 * j));
            const float val = dl * ((float)g + delta);
            s += val * (float)xb[e * 32 + lane];
        }
        return s;
    }
};

// ----------------------------------------------------------------------------------------------
// int8-dp4a DECODE for the IQ codebook quants (IQ2_XXS/XS/S, IQ3_XXS/S, IQ1_S/M). The int8-SIMD twin
// of the scalar `qdec<DW_IQ*>` above: instead of decoding each grid coordinate to f32 and doing float
// MACs (COMPUTE-bound on this APU), the grid coords land in int8 and `v_dot4` (hip_dp4a) dots them
// against the once-q8_1-quantized activation -- llama.cpp's `vec_dot_iq*_q8_1` technique. These
// specialize the SAME `qdp4a<WTYPE>` / `qdp4a_traits<WTYPE>` the K-quants use, so they ride the SAME
// `qmatvec_dp4a_core` / `moe_qmatvec_dp4a_core` + the SAME DEFINE_QMATVEC_DP4A launcher -- NO new core,
// NO new kernel. They sit HERE (not with the K-quant qdp4a<> at the top) because they index the
// __constant__ grid tables defined just above, exactly like the scalar qdec<DW_IQ*> twins.
//
// NSUB = 8: the work-unit is the 32-elem sub-block (sub = e in [0,8)), matching the q8_1 32-block
// scale `xd8[e]` -- so `partial(blk,xq8,xd8,e)` returns sub-block e's COMPLETE f32 contribution and
// block() = sum over e. Each sub-block has 4 groups l in [0,4) of 8 coords; the 8 coords split into a
// low int (coords 0..3) and high int (coords 4..7), each dotted against its q8_1 int (u[l*2], u[l*2+1]).
//
// SIGNS without CUDA's __vsub4/__vcmpne4 (absent on ROCm 7.13): the IQ2/IQ3 grids store POSITIVE
// magnitudes (verified <=62, valid signed int8) + a per-coord sign bit. By linearity
//   sum_i (-1)^s_i * g_i * u_i = sum_i g_i u_i - 2 * sum_{s_i set} g_i u_i
// so `dp4a_signed` = dp4a(g,u) - 2*dp4a(g & bytemask(s), u), where `bytemask` expands a 4-bit sign
// nibble to 0xFF per signed byte. No per-byte subtract needed. IQ1 grids store SIGNED coords already
// (no sign field) -> direct dp4a + a `sum(u)` term for the fractional `+delta` bias.

// Expand the low 4 bits of `sgn` to a per-byte mask: byte i = 0xFF if bit i set, else 0x00. Spread
// bits {0,1,2,3} to byte LSBs {0,8,16,24} (each 0 or 1), then *0xFF to fill the byte (no carry: 1*255).
__device__ __forceinline__ unsigned iq_bytemask4(unsigned sgn) {
    const unsigned s = (sgn & 1u) | ((sgn & 2u) << 7) | ((sgn & 4u) << 14) | ((sgn & 8u) << 21);
    return s * 0xFFu;
}

// Signed-grid 4-wide int8 dot: sum_i (-1)^sgn_i * g_i * u_i for the 4 magnitude bytes packed in `grid`
// (each in [0,127]) against the q8_1 int `u`, signs from the low 4 bits of `sgn`. dp4a-only (above).
__device__ __forceinline__ int dp4a_signed(int grid, int u, unsigned sgn) {
    const int gm = (int)((unsigned)grid & iq_bytemask4(sgn));
    return hip_dp4a(grid, u, 0) - 2 * hip_dp4a(gm, u, 0);
}

// IQ2_XXS (SYM 2.06bpw). Mirrors qdec<DW_IQ2_XXS>: per sub-block e, db = d*(0.5+(aux32_1>>28))*0.25
// is constant across all 32 coords; grid byte j of IQ2XXS_GRID_D[gi] is the magnitude, ksigns the sign.
template <> struct qdp4a<DW_IQ2_XXS> {
    static constexpr int NSUB = 8;
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int e) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint16_t* qs = reinterpret_cast<const uint16_t*>(blk + 2);
        const uint32_t aux32_0 = (uint32_t)qs[4 * e] | ((uint32_t)qs[4 * e + 1] << 16);
        const uint32_t aux32_1 = (uint32_t)qs[4 * e + 2] | ((uint32_t)qs[4 * e + 3] << 16);
        const float db = d * (0.5f + (float)(aux32_1 >> 28)) * 0.25f;
        const float d8 = __half2float(xd8[e]);
        const int* u = reinterpret_cast<const int*>(xq8) + e * 8;
        int idot = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int gi = (int)((aux32_0 >> (8 * l)) & 0xFF);
            const uint64_t entry = IQ2XXS_GRID_D[gi];
            const uint32_t signs = KSIGNS_IQ2XS_D[(aux32_1 >> (7 * l)) & 127];
            idot += dp4a_signed((int)(uint32_t)entry, u[2 * l], signs & 0xF);
            idot += dp4a_signed((int)(uint32_t)(entry >> 32), u[2 * l + 1], (signs >> 4) & 0xF);
        }
        return db * d8 * (float)idot;
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < NSUB; ++e) s += partial(blk, xq8, xd8, e);
        return s;
    }
};

// IQ2_XS (SYM 2.31bpw). Mirrors qdec<DW_IQ2_XS>: scale nibble per sub-block e is sc&0xF for groups
// 0,1 (db_lo) and sc>>4 for groups 2,3 (db_hi) -- two partial dots scaled separately (llama ls0/ls1).
template <> struct qdp4a<DW_IQ2_XS> {
    static constexpr int NSUB = 8;
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int e) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint16_t* qs = reinterpret_cast<const uint16_t*>(blk + 2);
        const uint8_t sc = blk[66 + e];
        const float db_lo = d * (0.5f + (float)(sc & 0xF)) * 0.25f;
        const float db_hi = d * (0.5f + (float)(sc >> 4)) * 0.25f;
        const float d8 = __half2float(xd8[e]);
        const int* u = reinterpret_cast<const int*>(xq8) + e * 8;
        int idot_lo = 0, idot_hi = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint16_t qv = qs[4 * e + l];
            const uint64_t entry = IQ2XS_GRID_D[qv & 511];
            const uint32_t signs = KSIGNS_IQ2XS_D[qv >> 9];
            const int part = dp4a_signed((int)(uint32_t)entry, u[2 * l], signs & 0xF)
                           + dp4a_signed((int)(uint32_t)(entry >> 32), u[2 * l + 1], (signs >> 4) & 0xF);
            if (l < 2) idot_lo += part; else idot_hi += part;
        }
        return d8 * (db_lo * (float)idot_lo + db_hi * (float)idot_hi);
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < NSUB; ++e) s += partial(blk, xq8, xd8, e);
        return s;
    }
};

// IQ2_S (SYM 2.56bpw). Mirrors qdec<DW_IQ2_S>: idx = qs[4e+l] | ((qh[e]<<(8-2l))&0x300), signs are a
// raw byte qs[32+4e+l] (NOT ksigns), scale nibble per sub-block like IQ2_XS (db_lo / db_hi).
template <> struct qdp4a<DW_IQ2_S> {
    static constexpr int NSUB = 8;
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int e) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint8_t* qs = blk + 2;
        const uint8_t* qh = blk + 66;
        const uint8_t sc = blk[74 + e];
        const float db_lo = d * (0.5f + (float)(sc & 0xF)) * 0.25f;
        const float db_hi = d * (0.5f + (float)(sc >> 4)) * 0.25f;
        const float d8 = __half2float(xd8[e]);
        const int* u = reinterpret_cast<const int*>(xq8) + e * 8;
        int idot_lo = 0, idot_hi = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int idx = (int)qs[4 * e + l] | (((int)qh[e] << (8 - 2 * l)) & 0x300);
            const uint64_t entry = IQ2S_GRID_D[idx];
            const uint32_t signs = qs[32 + 4 * e + l];
            const int part = dp4a_signed((int)(uint32_t)entry, u[2 * l], signs & 0xF)
                           + dp4a_signed((int)(uint32_t)(entry >> 32), u[2 * l + 1], (signs >> 4) & 0xF);
            if (l < 2) idot_lo += part; else idot_hi += part;
        }
        return d8 * (db_lo * (float)idot_lo + db_hi * (float)idot_hi);
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < NSUB; ++e) s += partial(blk, xq8, xd8, e);
        return s;
    }
};

// IQ3_XXS (SYM 3.06bpw). Mirrors qdec<DW_IQ3_XXS>: db = d*(0.5+(aux32>>28))*0.5 const per sub-block;
// each group l has TWO u32 grid entries (coords 0..3 from qs[8e+2l], 4..7 from qs[8e+2l+1]), ksigns.
template <> struct qdp4a<DW_IQ3_XXS> {
    static constexpr int NSUB = 8;
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int e) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint8_t* qs = blk + 2;
        const uint8_t* ss = blk + 2 + 64;
        const uint32_t aux32 = (uint32_t)ss[4 * e] | ((uint32_t)ss[4 * e + 1] << 8)
                             | ((uint32_t)ss[4 * e + 2] << 16) | ((uint32_t)ss[4 * e + 3] << 24);
        const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
        const float d8 = __half2float(xd8[e]);
        const int* u = reinterpret_cast<const int*>(xq8) + e * 8;
        int idot = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t signs = KSIGNS_IQ2XS_D[(aux32 >> (7 * l)) & 127];
            idot += dp4a_signed((int)IQ3XXS_GRID_D[qs[8 * e + 2 * l]], u[2 * l], signs & 0xF);
            idot += dp4a_signed((int)IQ3XXS_GRID_D[qs[8 * e + 2 * l + 1]], u[2 * l + 1], (signs >> 4) & 0xF);
        }
        return db * d8 * (float)idot;
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < NSUB; ++e) s += partial(blk, xq8, xd8, e);
        return s;
    }
};

// IQ3_S (SYM 3.31bpw). Mirrors qdec<DW_IQ3_S>: db = d*(1+2*nib) const per sub-block (nib from
// scales[e>>1]); idx high bit from qh with shift 8-2l (coords 0..3) / 7-2l (coords 4..7); raw sign byte.
template <> struct qdp4a<DW_IQ3_S> {
    static constexpr int NSUB = 8;
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int e) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint8_t* qs = blk + 2;
        const uint8_t* qh = blk + 66;
        const uint8_t* signs_b = blk + 74;
        const uint8_t* scales = blk + 106;
        const uint8_t sc = scales[e >> 1];
        const int nib = (e & 1) ? (sc >> 4) : (sc & 0xF);
        const float db = d * (1.0f + 2.0f * (float)nib);
        const float d8 = __half2float(xd8[e]);
        const int* u = reinterpret_cast<const int*>(xq8) + e * 8;
        int idot = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int idx0 = (int)qs[8 * e + 2 * l] | (((int)qh[e] << (8 - 2 * l)) & 256);
            const int idx1 = (int)qs[8 * e + 2 * l + 1] | (((int)qh[e] << (7 - 2 * l)) & 256);
            const uint32_t signs = signs_b[4 * e + l];
            idot += dp4a_signed((int)IQ3S_GRID_D[idx0], u[2 * l], signs & 0xF);
            idot += dp4a_signed((int)IQ3S_GRID_D[idx1], u[2 * l + 1], (signs >> 4) & 0xF);
        }
        return db * d8 * (float)idot;
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < NSUB; ++e) s += partial(blk, xq8, xd8, e);
        return s;
    }
};

// IQ1_S (1.56bpw). Mirrors qdec<DW_IQ1_S>: grid coords are SIGNED int8 already (no sign field) -> direct
// dp4a; dl = d*(2*((qh>>12)&7)+1), delta = +/-IQ1S_DELTA per sub-block. val = dl*(g+delta), so the
// fractional bias rides a sum_u term: result = dl*d8*(sum g*u + delta*sum u).
template <> struct qdp4a<DW_IQ1_S> {
    static constexpr int NSUB = 8;
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int e) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint8_t* qs = blk + 2;
        const uint16_t* qh = reinterpret_cast<const uint16_t*>(blk + 34);
        const uint16_t qhv = qh[e];
        const float dl = d * (2.0f * (float)((qhv >> 12) & 7) + 1.0f);
        const float delta = (qhv & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;
        const float d8 = __half2float(xd8[e]);
        const int* u = reinterpret_cast<const int*>(xq8) + e * 8;
        int idot = 0, isum = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int idx = (int)qs[e * 4 + l] | ((((int)(qhv >> (3 * l))) & 7) << 8);
            const uint64_t entry = IQ1S_GRID_D[idx];
            idot = hip_dp4a((int)(uint32_t)entry, u[2 * l], idot);
            idot = hip_dp4a((int)(uint32_t)(entry >> 32), u[2 * l + 1], idot);
            isum = hip_dp4a(0x01010101, u[2 * l], isum);
            isum = hip_dp4a(0x01010101, u[2 * l + 1], isum);
        }
        return dl * d8 * ((float)idot + delta * (float)isum);
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < NSUB; ++e) s += partial(blk, xq8, xd8, e);
        return s;
    }
};

// IQ1_M (1.75bpw). Mirrors qdec<DW_IQ1_M>: signed grid (direct dp4a); super-scale d from 4 scale-nibble
// tops; dl1 (groups 0,1) / dl2 (groups 2,3) from 3-bit fields; idx high bits + delta sign per group from
// qh. delta varies PER group -> accumulate dl*(sum g*u + delta*sum u) in f32 per group.
template <> struct qdp4a<DW_IQ1_M> {
    static constexpr int NSUB = 8;
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int e) {
        const uint8_t* qs = blk;
        const uint8_t* qh = blk + 32;
        const uint8_t* scales = blk + 48;
        const uint16_t sc0 = (uint16_t)scales[0] | ((uint16_t)scales[1] << 8);
        const uint16_t sc1 = (uint16_t)scales[2] | ((uint16_t)scales[3] << 8);
        const uint16_t sc2 = (uint16_t)scales[4] | ((uint16_t)scales[5] << 8);
        const uint16_t sc3 = (uint16_t)scales[6] | ((uint16_t)scales[7] << 8);
        const uint16_t scale_u16 = (uint16_t)((sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000));
        const float d = __half2float(__ushort_as_half(scale_u16));
        const uint16_t sc[4] = {sc0, sc1, sc2, sc3};
        const uint16_t scv = sc[e >> 1];
        const float dl1 = d * (2.0f * (float)((scv >> (6 * (e & 1)))     & 0x7) + 1.0f);
        const float dl2 = d * (2.0f * (float)((scv >> (6 * (e & 1) + 3)) & 0x7) + 1.0f);
        const uint8_t qh0 = qh[e * 2];
        const uint8_t qh1 = qh[e * 2 + 1];
        const float d8 = __half2float(xd8[e]);
        const int* u = reinterpret_cast<const int*>(xq8) + e * 8;
        float fs = 0.0f;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            int idx; float delta; float dl;
            if (l == 0)      { idx = (int)qs[e * 4]     | (((int)qh0 << 8) & 0x700); delta = (qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA; dl = dl1; }
            else if (l == 1) { idx = (int)qs[e * 4 + 1] | (((int)qh0 << 4) & 0x700); delta = (qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA; dl = dl1; }
            else if (l == 2) { idx = (int)qs[e * 4 + 2] | (((int)qh1 << 8) & 0x700); delta = (qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA; dl = dl2; }
            else             { idx = (int)qs[e * 4 + 3] | (((int)qh1 << 4) & 0x700); delta = (qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA; dl = dl2; }
            const uint64_t entry = IQ1S_GRID_D[idx];
            int idot = hip_dp4a((int)(uint32_t)entry, u[2 * l], 0);
            idot = hip_dp4a((int)(uint32_t)(entry >> 32), u[2 * l + 1], idot);
            int isum = hip_dp4a(0x01010101, u[2 * l], 0);
            isum = hip_dp4a(0x01010101, u[2 * l + 1], isum);
            fs += dl * ((float)idot + delta * (float)isum);
        }
        return d8 * fs;
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < NSUB; ++e) s += partial(blk, xq8, xd8, e);
        return s;
    }
};

// IQ dp4a byte strides (= on-disk super-block bytes; mirror qdw_traits::BYTES). All 256-elem.
template <> struct qdp4a_traits<DW_IQ2_XXS> { static constexpr int BYTES = 66;  };
template <> struct qdp4a_traits<DW_IQ2_XS>  { static constexpr int BYTES = 74;  };
template <> struct qdp4a_traits<DW_IQ2_S>   { static constexpr int BYTES = 82;  };
template <> struct qdp4a_traits<DW_IQ3_XXS> { static constexpr int BYTES = 98;  };
template <> struct qdp4a_traits<DW_IQ3_S>   { static constexpr int BYTES = 110; };
template <> struct qdp4a_traits<DW_IQ1_S>   { static constexpr int BYTES = 50;  };
template <> struct qdp4a_traits<DW_IQ1_M>   { static constexpr int BYTES = 56;  };

// IQ dp4a entry points (decode + MoE, f16/bf16/f32) via the SAME launcher generator as the K-quants.
DEFINE_QMATVEC_DP4A(iq2xxs, DW_IQ2_XXS)
DEFINE_QMATVEC_DP4A(iq2xs,  DW_IQ2_XS)
DEFINE_QMATVEC_DP4A(iq2s,   DW_IQ2_S)
DEFINE_QMATVEC_DP4A(iq3xxs, DW_IQ3_XXS)
DEFINE_QMATVEC_DP4A(iq3s,   DW_IQ3_S)
DEFINE_QMATVEC_DP4A(iq1_s,  DW_IQ1_S)
DEFINE_QMATVEC_DP4A(iq1_m,  DW_IQ1_M)

#undef DEFINE_QMATVEC_DP4A

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
DEFINE_QMATVECU(q2k,   DW_Q2_K)
DEFINE_QMATVECU(q3k,   DW_Q3_K)
DEFINE_QMATVECU(q5k,   DW_Q5_K)
DEFINE_QMATVECU(q4_1,  DW_Q4_1)
DEFINE_QMATVECU(q5_0,  DW_Q5_0)
DEFINE_QMATVECU(q5_1,  DW_Q5_1)
DEFINE_QMATVECU(q8_1,  DW_Q8_1)
DEFINE_QMATVECU(iq2xxs, DW_IQ2_XXS)
DEFINE_QMATVECU(iq2xs,  DW_IQ2_XS)
DEFINE_QMATVECU(iq2s,   DW_IQ2_S)
DEFINE_QMATVECU(iq3xxs, DW_IQ3_XXS)
DEFINE_QMATVECU(iq3s,   DW_IQ3_S)
DEFINE_QMATVECU(iq4nl, DW_IQ4_NL)
DEFINE_QMATVECU(tq1_0, DW_TQ1_0)
DEFINE_QMATVECU(iq1_s, DW_IQ1_S)
DEFINE_QMATVECU(iq1_m, DW_IQ1_M)
DEFINE_QMATVECU(mxfp4,  DW_MXFP4)

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
DEFINE_MOE_QMATVECU(q2k,   DW_Q2_K)
DEFINE_MOE_QMATVECU(q3k,   DW_Q3_K)
DEFINE_MOE_QMATVECU(q5k,   DW_Q5_K)
DEFINE_MOE_QMATVECU(q4_1,  DW_Q4_1)
DEFINE_MOE_QMATVECU(q5_0,  DW_Q5_0)
DEFINE_MOE_QMATVECU(q5_1,  DW_Q5_1)
DEFINE_MOE_QMATVECU(q8_1,  DW_Q8_1)
DEFINE_MOE_QMATVECU(iq2xxs, DW_IQ2_XXS)
DEFINE_MOE_QMATVECU(iq2xs,  DW_IQ2_XS)
DEFINE_MOE_QMATVECU(iq2s,   DW_IQ2_S)
DEFINE_MOE_QMATVECU(iq3xxs, DW_IQ3_XXS)
DEFINE_MOE_QMATVECU(iq3s,   DW_IQ3_S)
DEFINE_MOE_QMATVECU(iq4nl, DW_IQ4_NL)
DEFINE_MOE_QMATVECU(tq1_0, DW_TQ1_0)
DEFINE_MOE_QMATVECU(iq1_s, DW_IQ1_S)
DEFINE_MOE_QMATVECU(iq1_m, DW_IQ1_M)
DEFINE_MOE_QMATVECU(mxfp4,  DW_MXFP4)

#undef DEFINE_MOE_QMATVECU
#undef DEFINE_QMATVECU
#undef DW_Q8_0
#undef DW_Q4_0
#undef DW_Q4_K
#undef DW_Q6_K
#undef DW_IQ4_XS
#undef DW_TQ2_0
#undef DW_Q2_K
#undef DW_Q3_K
#undef DW_Q5_K
#undef DW_Q4_1
#undef DW_Q5_0
#undef DW_Q5_1
#undef DW_Q8_1
#undef DW_IQ2_XXS
#undef DW_IQ2_XS
#undef DW_IQ2_S
#undef DW_IQ3_XXS
#undef DW_IQ3_S
#undef DW_IQ4_NL
#undef DW_TQ1_0
#undef DW_IQ1_S
#undef DW_IQ1_M

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
#define WT_Q5_K   6   // 176 B, 256 elems, ASYM super-block.    q_w = nib|hi<<4(0..31); scale d*sc, min dmin*m
#define WT_Q4_1   7   // 20 B,  32 elems,  ASYM legacy.         q_w = nib(0..15);   scale d, min -m
#define WT_Q5_0   8   // 22 B,  32 elems,  SYM legacy 5-bit.    q_w = (nib|hi<<4)-16;   scale d
#define WT_Q5_1   9   // 24 B,  32 elems,  ASYM legacy 5-bit.   q_w = nib|hi<<4(0..31); scale d, min -m
#define WT_Q8_1   10  // 36 B,  32 elems,  SYM 8-bit (+sum).    q_w = qs[k];        scale d
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
template <> struct wt_traits<WT_Q5_K>   { static constexpr int BYTES = 176; static constexpr int SBLK_ELEMS = 256; static constexpr bool SYMMETRIC = false; static constexpr int SCALE_STEP = 32; };
template <> struct wt_traits<WT_Q4_1>   { static constexpr int BYTES = 20;  static constexpr int SBLK_ELEMS = 32;  static constexpr bool SYMMETRIC = false; static constexpr int SCALE_STEP = 32; };
template <> struct wt_traits<WT_Q5_0>   { static constexpr int BYTES = 22;  static constexpr int SBLK_ELEMS = 32;  static constexpr bool SYMMETRIC = true;  static constexpr int SCALE_STEP = 32; };
template <> struct wt_traits<WT_Q5_1>   { static constexpr int BYTES = 24;  static constexpr int SBLK_ELEMS = 32;  static constexpr bool SYMMETRIC = false; static constexpr int SCALE_STEP = 32; };
template <> struct wt_traits<WT_Q8_1>   { static constexpr int BYTES = 36;  static constexpr int SBLK_ELEMS = 32;  static constexpr bool SYMMETRIC = true;  static constexpr int SCALE_STEP = 32; };

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
    // SWAR nibble extract: one 128-bit load of the 16 packed bytes, then  (w >> shift) & 0x0F0F0F0F
    // across all 4 dwords -- 4 lane-parallel ANDs replace 16 per-byte v_and_b16 ops. q_w is the
    // UNSIGNED nibble 0..15 (the min term carries the offset), so no sign-extend is needed.
    uint32_t w[4]; __builtin_memcpy(w, qc, 16);
    v4i_ld v;
    #pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = (int)((w[j] >> shift) & 0x0F0F0F0Fu);
    return v;
}

// Q6_K: super-block = ql[128], qh[64], scales[16] (signed i8), d (f16). 32-sub-block g covers the
// SAME quadrant layout as qdec<DW_Q6_K>: half=g/4 of the block, quad=g%4. q_w = q-32 (signed,
// [-32,31]). k=half..half+15 within the sub-block (this routine returns one 16-half).
template <>
__device__ __forceinline__ v4i_ld decode_w_half<WT_Q6_K>(const uint8_t* sb, int g, int half) {
    const uint8_t* qh_b = sb + 128;
    const int hf = g >> 2;           // 0,1  (which 128-half)
    const int quad = g & 3;          // 0..3
    // ql[kk] for quad 0/2, ql[kk+32] for quad 1/3 -- both 16 contiguous bytes (kk=half..half+15).
    const uint8_t* ql = sb + 64 * hf + ((quad & 1) ? 32 : 0) + half;
    const uint8_t* qh = qh_b + 32 * hf + half;
    const int lshift = (quad >> 1) ? 4 : 0;     // low nibble (q0/q1) vs high nibble (q2/q3)
    const int hshift = quad * 2;                 // qh 2-bit field: bits 0,2,4,6 for quad 0..3
    // SWAR over 4 dwords: load the 16 ql + 16 qh bytes as two 128-bit vectors, extract the 6-bit
    // weight lane-parallel (nibble | (2-bit<<4)) -- 8 dword ops replace 16x{and_b16,lshr_b16,or}.
    // Then center to signed [-32,31] borrow-free: ((e|0x80808080) - 0x20202020) ^ 0x80808080
    // (e in [0,63]; OR sets bit7 so the per-byte subtract never borrows; final XOR yields (int8)e-32).
    uint32_t lw[4], hw[4]; __builtin_memcpy(lw, ql, 16); __builtin_memcpy(hw, qh, 16);
    v4i_ld v;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t lo = (lw[j] >> lshift) & 0x0F0F0F0Fu;
        const uint32_t hi = ((hw[j] >> hshift) & 0x03030303u) << 4;
        const uint32_t e  = lo | hi;                              // unsigned 6-bit, per byte [0,63]
        v[j] = (int)((((e | 0x80808080u) - 0x20202020u) ^ 0x80808080u)); // signed (e-32) per byte
    }
    return v;
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

// Q5_K: super-block = f16 d, f16 dmin, 12 scale bytes, qh[32], qs[128]. Sub-block g (0..8): chunk
// c=g/2 of qs (byte c*32+k), even g = LOW nibble / odd g = HIGH nibble, 5th bit = qh[k] bit g. q_w =
// the UNSIGNED 5-bit code nib|hi<<4 in [0,31] (the min term carries the offset). Mirrors Q4_K + qh.
template <>
__device__ __forceinline__ v4i_ld decode_w_half<WT_Q5_K>(const uint8_t* sb, int g, int half) {
    const uint8_t* qh = sb + 16;
    const uint8_t* qs = sb + 48;
    const int chunk = g >> 1;
    const int shift = (g & 1) ? 4 : 0;
    const uint8_t* qc = qs + chunk * 32 + half;  // k=half..half+15 of this sub-block
    int8_t tmp[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int l   = half + i;
        const int nib = (qc[i] >> shift) & 0x0F;
        const int bit = (qh[l] >> g) & 1;
        tmp[i] = (int8_t)(nib | (bit << 4));     // 0..31
    }
    v4i_ld v; __builtin_memcpy(&v, tmp, 16); return v;
}

// Q4_1: f16 d, f16 m, 16 nibble-pairs at +4 (SUBS=1, g==0). half=0 -> k 0..15 = LOW nibbles, half=16
// -> HIGH. q_w = the UNSIGNED nibble 0..15 (the +m bias rides decode_w_secscale).
template <>
__device__ __forceinline__ v4i_ld decode_w_half<WT_Q4_1>(const uint8_t* sb, int, int half) {
    const uint8_t* qs = sb + 4;
    const int shift = (half == 0) ? 0 : 4;
    int8_t tmp[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) tmp[i] = (int8_t)((qs[i] >> shift) & 0x0F);
    v4i_ld v; __builtin_memcpy(&v, tmp, 16); return v;
}

// Q5_0: f16 d, qh[4] (u32 LE) at +2, 16 nibble-pairs at +6 (SUBS=1). half=0 -> k 0..15 (low nibbles,
// qh bits 0..15), half=16 -> k 16..31 (high nibbles, qh bits 16..31). q_w = (nib|bit<<4)-16 (signed
// [-16,15]). qh read byte-wise (22 B stride leaves sb+2 only 2-aligned).
template <>
__device__ __forceinline__ v4i_ld decode_w_half<WT_Q5_0>(const uint8_t* sb, int, int half) {
    const uint8_t* qhb = sb + 2;
    const uint32_t qh = (uint32_t)qhb[0] | ((uint32_t)qhb[1] << 8) | ((uint32_t)qhb[2] << 16) | ((uint32_t)qhb[3] << 24);
    const uint8_t* qs = sb + 6;
    const int shift = (half == 0) ? 0 : 4;
    int8_t tmp[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int nib = (qs[i] >> shift) & 0x0F;
        const int bit = (int)((qh >> (half + i)) & 1u);
        tmp[i] = (int8_t)((nib | (bit << 4)) - 16);
    }
    v4i_ld v; __builtin_memcpy(&v, tmp, 16); return v;
}

// Q5_1: f16 d, f16 m, qh[4] (u32 LE) at +4, 16 nibble-pairs at +8 (SUBS=1). Like Q5_0 but q_w = the
// UNSIGNED 5-bit code nib|bit<<4 in [0,31] (NOT centered; the +m bias rides decode_w_secscale).
template <>
__device__ __forceinline__ v4i_ld decode_w_half<WT_Q5_1>(const uint8_t* sb, int, int half) {
    const uint8_t* qhb = sb + 4;
    const uint32_t qh = (uint32_t)qhb[0] | ((uint32_t)qhb[1] << 8) | ((uint32_t)qhb[2] << 16) | ((uint32_t)qhb[3] << 24);
    const uint8_t* qs = sb + 8;
    const int shift = (half == 0) ? 0 : 4;
    int8_t tmp[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int nib = (qs[i] >> shift) & 0x0F;
        const int bit = (int)((qh >> (half + i)) & 1u);
        tmp[i] = (int8_t)(nib | (bit << 4));
    }
    v4i_ld v; __builtin_memcpy(&v, tmp, 16); return v;
}

// Q8_1: 32 int8 quants at +4 (SUBS=1, g==0); the 16B half is a straight load (Q8_0 with the qs
// shifted past the extra f16 sum field). q_w = qs[k].
template <>
__device__ __forceinline__ v4i_ld decode_w_half<WT_Q8_1>(const uint8_t* sb, int, int half) {
    v4i_ld v;
    __builtin_memcpy(&v, sb + 4 + half, 16);
    return v;
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
// Q5_K: d * sc_g (sc from get_scale_min_k4 for sub-block g = sub16/2) -- same scale layout as Q4_K.
template <> __device__ __forceinline__ float decode_w_mainscale<WT_Q5_K>(const uint8_t* sb, int sub16) {
    const float d = __half2float(*reinterpret_cast<const __half*>(sb));
    int sc, m; q4k_scale_min(sb + 4, sub16 >> 1, &sc, &m);
    return d * (float)sc;
}
// Q4_1/Q5_0/Q5_1/Q8_1: single per-block f16 scale `d` at byte 0.
template <> __device__ __forceinline__ float decode_w_mainscale<WT_Q4_1>(const uint8_t* sb, int) {
    return __half2float(*reinterpret_cast<const __half*>(sb));
}
template <> __device__ __forceinline__ float decode_w_mainscale<WT_Q5_0>(const uint8_t* sb, int) {
    return __half2float(*reinterpret_cast<const __half*>(sb));
}
template <> __device__ __forceinline__ float decode_w_mainscale<WT_Q5_1>(const uint8_t* sb, int) {
    return __half2float(*reinterpret_cast<const __half*>(sb));
}
template <> __device__ __forceinline__ float decode_w_mainscale<WT_Q8_1>(const uint8_t* sb, int) {
    return __half2float(*reinterpret_cast<const __half*>(sb));
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
// Q5_K: dmin * m_g (same min layout as Q4_K).
template <> __device__ __forceinline__ float decode_w_secscale<WT_Q5_K>(const uint8_t* sb, int g) {
    const float dmin = __half2float(*reinterpret_cast<const __half*>(sb + 2));
    int sc, m; q4k_scale_min(sb + 4, g, &sc, &m);
    return dmin * (float)m;
}
// Q4_1/Q5_1: the legacy +m bias. qmmq accumulates out -= d_x*ix*secscale and the true value adds
// +m*d_x*ix, so secscale = -m (m = the f16 at sb+2).
template <> __device__ __forceinline__ float decode_w_secscale<WT_Q4_1>(const uint8_t* sb, int) {
    return -__half2float(*reinterpret_cast<const __half*>(sb + 2));
}
template <> __device__ __forceinline__ float decode_w_secscale<WT_Q5_1>(const uint8_t* sb, int) {
    return -__half2float(*reinterpret_cast<const __half*>(sb + 2));
}
template <> __device__ __forceinline__ float decode_w_secscale<WT_Q5_0>(const uint8_t*, int) { return 0.0f; }
template <> __device__ __forceinline__ float decode_w_secscale<WT_Q8_1>(const uint8_t*, int) { return 0.0f; }

// Shared int8 tile row stride: 32 data bytes + 16 pad bytes = 48. The pad is 16-ALIGNED so the 32-byte
// fragment loads (`__builtin_memcpy(&frag, &sXi[row*SROW], 32)`) lower to `ds_load_b128` (two 16-byte
// LDS reads) instead of byte-strided `ds_load_2addr_b32` -- a 36-byte stride is not 16-aligned so the
// per-lane base broke b128 vectorization. Measured +11% on Q4_K prefill (gfx1151, oracle nbad=0). The
// 16-byte pad also keeps consecutive rows off the same LDS bank-group (bank-conflict-free, profiled
// LDSBankConflict ~0.03%). 26624 B LDS total (2*128*48 + scales) -> still fits occupancy (10 waves).
#define SROW 48
// Map a LOCAL tile row LR (0..127) to the M-index GM into xq/xd/xs and validity VALID. DENSE: GM is
// the absolute row (row_tile + LR), valid while in [0,M). MOE: the tile is one expert's slice, so LR
// indexes the expert-sorted positions [pos0, pos0+nrows_tile); GM = slot_map[pos0+LR] gathers/scatters
// the ORIGINAL routed slot (xq read AND y write share it). `if constexpr (MOE)` elides one branch, so
// the dense path codegens byte-identical (GM == row_tile+LR, no slot_map load).
#define MROW(LR, GM, VALID)                                                                          \
    int GM; bool VALID;                                                                              \
    if constexpr (MOE) { (VALID) = ((LR) < nrows_tile); (GM) = (VALID) ? slot_map[pos0 + (LR)] : 0; }\
    else { (GM) = row_tile + (LR); (VALID) = ((GM) < M); }
// Stage the X (activation) int8 tile + the weight int8 tile (per-type decode) + the per-block
// scales for 32-block BLK into shared buffer BUF. Identical X-half staging to the proven path; the
// WEIGHT tile uses decode_w_half<WTYPE>(super-block, g, half) with g = the sub-block index. Scales
// staged once/block: sWd (main, indexed by 16-group via SROW-free 2-slot rows), sWm (secondary).
// THREAD-COUNT-AGNOSTIC staging: strided loops over the staging work-items so the SAME macro covers
// any block size (TILE_M=128 -> 512 thr, 64 -> 256, 32 -> 128). The TN=128 weight tile is always 256
// int8 items (128 cols x 2 halves) + 128 scale cols; the X tile is TM rows. For the TILE_M=128/512-thread
// case the loops run exactly once with i==t (weight i<256, X i>=256), byte-identical to the proven
// hardcoded split -- the dense path codegen is unchanged.
#define STGI(BUF, BLK)                                                                          \
    do {                                                                                        \
        const int sb_ = (BLK) / SUBS;            /* on-disk super-block index */                \
        const int g_  = (BLK) % SUBS;            /* sub-block within the super-block */          \
        for (int i = t; i < 256 + TM * 2; i += THREADS) {                                       \
            if (i < 256) {                                                                       \
                const int r = i >> 1; const int half = (i & 1) * 16;                            \
                const int gn = col_tile + r;                                                    \
                v4i_ld v = {0,0,0,0};                                                           \
                if (gn < N) {                                                                    \
                    v = decode_w_half<WTYPE>(wq_eff + ((size_t)gn * nsblk + sb_) * WBYTES, g_, half);\
                }                                                                               \
                __builtin_memcpy(&sWi[(BUF)][r * SROW + half], &v, 16);                          \
            } else {                                                                            \
                const int j = i - 256; const int r = j >> 1; const int half = (j & 1) * 16;     \
                MROW(r, gm, mvalid);                                                            \
                v4i_ld v = {0,0,0,0};                                                           \
                if (mvalid) {                                                                   \
                    v = *reinterpret_cast<const v4i_ld*>(                                       \
                            &xq[(size_t)gm * K + (BLK) * 32 + half]);                           \
                }                                                                               \
                __builtin_memcpy(&sXi[(BUF)][r * SROW + half], &v, 16);                          \
            }                                                                                   \
        }                                                                                        \
        /* Scales staged to shared ONCE per block (like llama's x_df/y_df): 128 weight cols + TM      \
           X rows. sWd = MAIN scale (the only scale for STEP=32; the 1st-16 scale for Q6_K). sWm =     \
           SECONDARY scalar (dmin*m for ASYM; the 2nd-16 scale for Q6_K) -- else a 1-slot dummy. sxs = \
           the q8_1 per-block activation sum (ASYM only). */                                          \
        for (int i = t; i < 128 + TM; i += THREADS) {                                           \
            if (i < 128) {                                                                       \
                const int gn = col_tile + i;                                                    \
                if (gn < N) {                                                                    \
                    const uint8_t* wb = wq_eff + ((size_t)gn * nsblk + sb_) * WBYTES;           \
                    sWd[(BUF)][i] = decode_w_mainscale<WTYPE>(wb, g_ * 2 + 0);                   \
                    if constexpr (HAS_SEC) sWm[(BUF)][i] = decode_w_secscale<WTYPE>(wb, g_);     \
                } else {                                                                        \
                    sWd[(BUF)][i] = 0.0f; if constexpr (HAS_SEC) sWm[(BUF)][i] = 0.0f;          \
                }                                                                               \
            } else {                                                                            \
                const int r = i - 128;                                                          \
                MROW(r, gm, mvalid);                                                            \
                sxd[(BUF)][r] = mvalid ? __half2float(xd[(size_t)gm * nsblk32 + (BLK)]) : 0.0f;  \
                if constexpr (!SYMM) {                                                           \
                    sxs[(BUF)][r] = mvalid ? (float)xs[(size_t)gm * nsblk32 + (BLK)] : 0.0f;     \
                }                                                                               \
            }                                                                                   \
        }                                                                                       \
    } while (0)

// Templated device core: identical for every weight type except the WTYPE-selected decode + the
// compile-time accumulation shape (SYMMETRIC / SCALE_STEP). `xs` is read iff !SYMMETRIC (the q8_1
// per-block activation sum from quantize_q8_1); symmetric types pass any pointer (never derefed).
// NWAVE_M = number of 32-row M-waves (TILE_M = NWAVE_M*32). The N tile is fixed at 128 (4 waves).
// NWAVE_M=4 -> the proven 128x128/512-thread tile (default; dense path). MoE picks NWAVE_M=2 (64-row)
// or 1 (32-row) when experts route few tokens, so the WMMA M-rows are filled instead of half/mostly
// empty (the routed avg is ~64 tok/expert at full prefill, far fewer per micro-batch). One block stays
// one expert's slice; smaller blocks co-reside more per CU (less LDS), recovering SIMD occupancy.
template <int WTYPE, bool MOE = false, int NWAVE_M = 4>
__device__ __forceinline__ void qmmq_core(
    const int M,
    const int N,
    const int K,            // multiple of 32
    const int ncol_tiles,   // ceil(N/128)
    const int8_t* __restrict__ xq,   // [M, K] int8 pre-quantized activations (MOE: [nslots, K])
    const __half* __restrict__ xd,   // [M, K/32] f16 per-32-block activation scales
    const uint8_t* __restrict__ wq,  // [N, K] weight super-blocks (MOE: [E,N,K] bank, expert-offset)
    __half* __restrict__ y,          // [M, N] (MOE: [nslots, N])
    const int* __restrict__ xs,      // [M, K/32] i32 per-32-block activation sum (ASYM only)
    const int* __restrict__ slot_map = nullptr,    // MOE: [nslots] expert-sorted pos -> original slot
    const int* __restrict__ tile_expert = nullptr, // MOE: [num_row_tiles] expert id per row-tile
    const int* __restrict__ tile_pos0 = nullptr,   // MOE: [num_row_tiles] first sorted pos in tile
    const int* __restrict__ tile_nrows = nullptr   // MOE: [num_row_tiles] valid rows in tile (<=128)
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
    constexpr int TM         = NWAVE_M * 32;       // M-tile rows
    constexpr int THREADS    = NWAVE_M * 128;      // 4 N-waves * 32 lanes * NWAVE_M M-waves
    constexpr int XSN        = SYMM ? 1 : TM;      // sxs slots/row (ASYM only)
    const int t = threadIdx.x;       // 0..THREADS-1
    const int lane = t & 31;
    const int warp = t >> 5;         // 0..(4*NWAVE_M-1)
    const int wave_m = warp % NWAVE_M;   // 0..NWAVE_M-1 (M sub-tile, x32)
    const int wave_n = warp / NWAVE_M;   // 0..3 (N sub-tile, x32)
    const int nsblk32 = K >> 5;          // # of 32-blocks along K (for xq/xd/xs indexing)
    const int nsblk   = K / SBLK_ELEMS;  // # of on-disk super-blocks along K (for wq indexing)
    const int nblk    = nsblk32;         // K-loop is over 32-blocks (WMMA granularity)
    // Tile -> output window. DENSE: blockIdx.x enumerates (row-tile x col-tile) over the one weight.
    // MOE: blockIdx.x enumerates (expert-row-tile x col-tile); each row-tile is one expert's <=128-token
    // slice (built host-side, never straddling an expert), so the weight base is offset to that expert
    // and rows are gathered/scattered via MROW. row_tile stays 0 for MOE (the slice offset is pos0).
    int row_tile, col_tile, pos0 = 0, nrows_tile = 0;
    const uint8_t* wq_eff;
    if constexpr (MOE) {
        const int rt = blockIdx.x / ncol_tiles;
        col_tile = (blockIdx.x % ncol_tiles) * 128;
        row_tile = 0;
        pos0 = tile_pos0[rt];
        nrows_tile = tile_nrows[rt];
        wq_eff = wq + (size_t)tile_expert[rt] * N * nsblk * WBYTES;
    } else {
        row_tile = (blockIdx.x / ncol_tiles) * TM;
        col_tile = (blockIdx.x % ncol_tiles) * 128;
        wq_eff = wq;
    }

    __shared__ __attribute__((aligned(16))) int8_t sXi[2][TM * SROW];
    __shared__ __attribute__((aligned(16))) int8_t sWi[2][128 * SROW];
    __shared__ float sxd[2][TM];      // per-row activation scale (proven Q8_0 layout)
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
    const int rbase = wave_m * 32;   // LOCAL tile-row base (row_tile folded into MROW for DENSE)
    const int cbase = col_tile + wave_n * 32;
    const int c0 = cbase + lcol;
    const int c1 = cbase + 16 + lcol;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int lr0 = rbase + i * 2 + lrow;
        const int lr1 = rbase + 16 + i * 2 + lrow;
        MROW(lr0, d0, v0);
        MROW(lr1, d1, v1);
        if (c0 < N) {
            if (v0) y[(size_t)d0 * N + c0] = __float2half(acc00[i]);
            if (v1) y[(size_t)d1 * N + c0] = __float2half(acc10[i]);
        }
        if (c1 < N) {
            if (v0) y[(size_t)d0 * N + c1] = __float2half(acc01[i]);
            if (v1) y[(size_t)d1 * N + c1] = __float2half(acc11[i]);
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
DEFINE_QMMQ(q5k,   WT_Q5_K)
DEFINE_QMMQ(q4_1,  WT_Q4_1)
DEFINE_QMMQ(q5_0,  WT_Q5_0)
DEFINE_QMMQ(q5_1,  WT_Q5_1)
DEFINE_QMMQ(q8_1,  WT_Q8_1)

#undef DEFINE_QMMQ

// FUSED indexed-MoE PREFILL GEMM: the SAME `qmmq_core<WTYPE,true>` tiled iu8 WMMA machine, but the
// routed tokens are grouped BY EXPERT host-side (slot_map/tile_* work-items) so each expert's weight
// is staged ONCE and amortized over all its tokens via the matrix cores -- llama's mul_mat_id strategy.
// This REPLACES the per-slot `moe_qmatvec_*` matvec (which re-read an expert's weight once per token,
// no matrix cores) on the prefill path. Decode (1 row) stays on the dp4a matvec. One WTYPE row adds a
// fused-MoE prefill type, exactly like the dense DEFINE_QMMQ above -- no new kernel body.
// Per (type, M-tile) entry. The launcher picks the M-tile by routed fill: `_f16` = TILE_M 128 (full
// prefill), `_tm64`/`_tm32` = NWAVE_M 2/1 for the few-token-per-expert micro-batches where a 128-row
// tile would leave most WMMA M-rows empty. The block dim (NWAVE_M*128) is set to match on the host side.
#define DEFINE_MOE_QMMQ_NW(NAME, WTYPE, SUF, NW)                                                 \
    extern "C" __global__ void moe_qmmq_##NAME##SUF(                                             \
        const int M, const int N, const int K, const int ncol_tiles,                             \
        const int8_t* __restrict__ xq, const __half* __restrict__ xd,                            \
        const uint8_t* __restrict__ wq, __half* __restrict__ y,                                  \
        const int* __restrict__ xs, const int* __restrict__ slot_map,                            \
        const int* __restrict__ tile_expert, const int* __restrict__ tile_pos0,                  \
        const int* __restrict__ tile_nrows) {                                                    \
        qmmq_core<WTYPE, true, NW>(M, N, K, ncol_tiles, xq, xd, wq, y, xs,                        \
            slot_map, tile_expert, tile_pos0, tile_nrows);                                       \
    }
#define DEFINE_MOE_QMMQ(NAME, WTYPE)                                                             \
    DEFINE_MOE_QMMQ_NW(NAME, WTYPE, _f16,      4)                                                 \
    DEFINE_MOE_QMMQ_NW(NAME, WTYPE, _tm64_f16, 2)                                                 \
    DEFINE_MOE_QMMQ_NW(NAME, WTYPE, _tm32_f16, 1)

DEFINE_MOE_QMMQ(q8_0,  WT_Q8_0)
DEFINE_MOE_QMMQ(q4_0,  WT_Q4_0)
DEFINE_MOE_QMMQ(q4k,   WT_Q4_K)
DEFINE_MOE_QMMQ(q6k,   WT_Q6_K)
DEFINE_MOE_QMMQ(iq4xs, WT_IQ4_XS)
DEFINE_MOE_QMMQ(tq2_0, WT_TQ2_0)
DEFINE_MOE_QMMQ(q5k,   WT_Q5_K)
DEFINE_MOE_QMMQ(q4_1,  WT_Q4_1)
DEFINE_MOE_QMMQ(q5_0,  WT_Q5_0)
DEFINE_MOE_QMMQ(q5_1,  WT_Q5_1)
DEFINE_MOE_QMMQ(q8_1,  WT_Q8_1)

#undef DEFINE_MOE_QMMQ
#undef DEFINE_MOE_QMMQ_NW
#undef MROW
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
#undef WT_Q5_K
#undef WT_Q4_1
#undef WT_Q5_0
#undef WT_Q5_1
#undef WT_Q8_1
#endif
