#include <metal_stdlib>
using namespace metal;

// Port of ggml-metal's `kernel_mul_mv_ext_q4_f32_impl` (the small-batch /
// 2..8 activation-column quantized mat-vec) specialised for Q8_0.
//
// The stock simdgroup-matrix GEMM (`kernel_mul_mm_q8_0_f32`) is compute-bound
// and pays for it at tiny batch sizes (e.g. the speculative-decode / MTP verify
// step, where the activation matrix has m = gamma+1 = 2..6 rows). This kernel
// reads each weight chunk exactly once and dots it against `R1PTG` activation
// columns, so the cost is dominated by streaming the weights (memory-bound),
// matching the m==1 mat-vec instead of the GEMM.
//
// Differences vs upstream:
//   * `r1ptg`/`nxpsg` are compile-time template params with distinct
//     `[[host_name]]` entry points (e.g. `kernel_mul_mv_ext_q8_0_f32_r1_2_nx8`)
//     rather than Metal function constants — our runtime-compile harness keys
//     pipelines by name, so per-variant entry points are the simplest route and
//     are exactly as fast (the values were uniform anyway).
//   * `nsg`/`ne12`/`r2`/`r3` are passed as ordinary kernel arguments (matching
//     the layout produced by `set_params!` in the Rust launcher) instead of a
//     packed kargs struct + function constants.
//
// Semantics (reverse ggml indexing, identical to call_quantized_matmul_mm_t):
//   src0 = weights [.., ne01 rows, ne00 cols], Q8_0-quantised, row stride nb01 bytes
//   src1 = activations [.., ne11 rows, ne10 cols], f32, row stride nb11 bytes
//   dst  = [.., ne1 rows, ne0 cols] f32
// Computes dst[i11, i01] = sum_k src1[i11, k] * dequant(src0[i01, k]).

#define QK8_0 32
typedef struct {
    half   d;          // delta
    int8_t qs[QK8_0];  // quants
} block_q8_0;

// Dequantize the `il`-th float4 chunk (4 contiguous quants) of a Q8_0 block.
// For a 32-element block there are 8 chunks (il in 0..8); 4*(il%4)+16*(il/4)
// reduces to 4*il for il<8 i.e. plain contiguous float4 reads.
static inline void dequantize_q8_0_t4(device const block_q8_0 * xb, short il, thread float4 & reg) {
    device const int8_t * qs = xb->qs;
    const float d = (float) xb->d;
    reg[0] = qs[4*(il%4) + 0 + 16*(il/4)] * d;
    reg[1] = qs[4*(il%4) + 1 + 16*(il/4)] * d;
    reg[2] = qs[4*(il%4) + 2 + 16*(il/4)] * d;
    reg[3] = qs[4*(il%4) + 3 + 16*(il/4)] * d;
}

// R1PTG : activation columns (src1 rows) processed per threadgroup row (2..5)
// NXPSG : threads along the K dimension per simdgroup (4 / 8 / 16)
template<short R1PTG, short NXPSG>
static inline void mul_mv_ext_q8_0_f32_impl(
        device const char  * src0,
        device const char  * src1,
        device       char  * dst,
        constant   int64_t & ne00,   // K (weight cols == activation cols)
        constant   int64_t & ne01,   // weight rows (output cols)
        constant  uint64_t & nb01,   // weight row stride (bytes)
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne11,   // activation rows (m)
        constant  uint64_t & nb11,   // activation row stride (bytes)
        constant  uint64_t & nb12,
        constant  uint64_t & nb13,
        constant   int64_t & ne0,    // dst cols (== ne01)
        constant   int64_t & ne1,    // dst rows (== ne11)
        constant   int     & nsg,    // simdgroups per threadgroup
        constant   int     & ne12,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3   tgpig,
        ushort  tiisg,
        ushort  sgitg) {
    // Q8_0: 32 elements/block, float4 chunks => 8 chunks per block.
    const short chpb = QK8_0 / 4; // 8
    const short chpt = 4;         // chunks per thread per iter (matches upstream)

    const short nypsg = 32 / NXPSG;   // weight rows handled per simdgroup
    const short tx = tiisg % NXPSG;   // lane index along K
    const short ty = tiisg / NXPSG;   // which weight row within the simdgroup

    const int i01 = tgpig.x * (nypsg * nsg) + nypsg * sgitg + ty; // weight row
    const int i11 = tgpig.y * R1PTG;                              // first activation row
    const int i1m = tgpig.z;                                      // batch (i12 + i13*ne12)

    const int i12 = i1m % ne12;
    const int i13 = i1m / ne12;

    const uint64_t offset0 = i01 * nb01 + (i12 / r2) * nb02 + (i13 / r3) * nb03;
    const uint64_t offset1 = i11 * nb11 + (i12) * nb12 + (i13) * nb13;

    // Each thread starts tx/chpb blocks in, then walks chunks of float4.
    device const block_q8_0 * xq = (i01 < ne01)
        ? (device const block_q8_0 *) (src0 + offset0) + tx / chpb
        : (device const block_q8_0 *) src0;

    device const float4 * y4[R1PTG];
    for (short ir1 = 0; ir1 < R1PTG; ++ir1) {
        y4[ir1] = (i11 + ir1 < ne11)
            ? (device const float4 *) (src1 + offset1 + ir1 * nb11) + tx
            : (device const float4 *) src1;
    }

    float sumf[R1PTG] = { 0.0f };

    short cch = tx % chpb; // current chunk index within the block

    for (int ich = tx; 4 * ich < ne00; ich += chpt * NXPSG) {
        float4 lx[chpt];

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
            dequantize_q8_0_t4(xq, cch, lx[ch]);

            cch += NXPSG;
            if (cch >= chpb) {
                xq  += cch / chpb;
                cch %= chpb;
            }
        }

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
#pragma unroll(R1PTG)
            for (short ir1 = 0; ir1 < R1PTG; ++ir1) {
                sumf[ir1] += dot(lx[ch], y4[ir1][ch * NXPSG]);
            }
        }

#pragma unroll(R1PTG)
        for (short ir1 = 0; ir1 < R1PTG; ++ir1) {
            y4[ir1] += chpt * NXPSG;
        }
    }

    // reduce across the NXPSG lanes that share a weight row
    for (short ir1 = 0; ir1 < R1PTG; ++ir1) {
        if (NXPSG >= 32) { sumf[ir1] += simd_shuffle_down(sumf[ir1], 16); }
        if (NXPSG >= 16) { sumf[ir1] += simd_shuffle_down(sumf[ir1],  8); }
        if (NXPSG >=  8) { sumf[ir1] += simd_shuffle_down(sumf[ir1],  4); }
        if (NXPSG >=  4) { sumf[ir1] += simd_shuffle_down(sumf[ir1],  2); }
        if (NXPSG >=  2) { sumf[ir1] += simd_shuffle_down(sumf[ir1],  1); }
    }

    if (tx == 0) {
        for (short ir1 = 0; ir1 < R1PTG && i11 + ir1 < ne11; ++ir1) {
            device float * dst_f32 = (device float *) dst + (uint64_t) i1m * ne0 * ne1 + (uint64_t)(i11 + ir1) * ne0;
            if (i01 < ne01) {
                dst_f32[i01] = sumf[ir1];
            }
        }
    }
}

#define MUL_MV_EXT_Q8_0(R1PTG, NXPSG)                                          \
kernel void kernel_mul_mv_ext_q8_0_f32_r1_##R1PTG##_nx##NXPSG(                 \
        device const char  * src0,                                            \
        device const char  * src1,                                            \
        device       char  * dst,                                             \
        constant   int64_t & ne00,                                            \
        constant   int64_t & ne01,                                            \
        constant  uint64_t & nb01,                                            \
        constant  uint64_t & nb02,                                            \
        constant  uint64_t & nb03,                                            \
        constant   int64_t & ne11,                                            \
        constant  uint64_t & nb11,                                            \
        constant  uint64_t & nb12,                                            \
        constant  uint64_t & nb13,                                            \
        constant   int64_t & ne0,                                             \
        constant   int64_t & ne1,                                             \
        constant   int     & nsg,                                             \
        constant   int     & ne12,                                            \
        constant   uint    & r2,                                              \
        constant   uint    & r3,                                              \
        uint3   tgpig[[threadgroup_position_in_grid]],                        \
        ushort  tiisg[[thread_index_in_simdgroup]],                           \
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {                    \
    mul_mv_ext_q8_0_f32_impl<R1PTG, NXPSG>(                                   \
        src0, src1, dst, ne00, ne01, nb01, nb02, nb03, ne11, nb11, nb12, nb13,\
        ne0, ne1, nsg, ne12, r2, r3, tgpig, tiisg, sgitg);                    \
}

// r1ptg in {2,3,4,5} (the dispatch maps ne11 in 2..8 onto these), each with
// the three nxpsg widths the dispatch can pick (4 / 8 / 16).
MUL_MV_EXT_Q8_0(2, 4)
MUL_MV_EXT_Q8_0(2, 8)
MUL_MV_EXT_Q8_0(2, 16)
MUL_MV_EXT_Q8_0(3, 4)
MUL_MV_EXT_Q8_0(3, 8)
MUL_MV_EXT_Q8_0(3, 16)
MUL_MV_EXT_Q8_0(4, 4)
MUL_MV_EXT_Q8_0(4, 8)
MUL_MV_EXT_Q8_0(4, 16)
MUL_MV_EXT_Q8_0(5, 4)
MUL_MV_EXT_Q8_0(5, 8)
MUL_MV_EXT_Q8_0(5, 16)
