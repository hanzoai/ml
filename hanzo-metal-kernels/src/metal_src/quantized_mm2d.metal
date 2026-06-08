// Metal 4 matmul2d-based Q8_0 mul_mm (prefill GEMM) path.
//
// This is a SEPARATE source unit from quantized.metal because it requires the
// Metal 4 cooperative tensor-ops API (mpp::tensor_ops::matmul2d), which only
// compiles when the library is built at MTLLanguageVersion 4.0. quantized.metal
// is compiled with the default (broadly-compatible) options and must keep
// working on pre-Metal-4 GPUs, so the matmul2d kernel is isolated here and only
// loaded on Metal-4-capable Apple GPUs (see kernels/quantized.rs dispatch gate).
//
// Modeled on llama.cpp's kernel_mul_mm matmul2d path
// (ggml/src/ggml-metal/ggml-metal.metal, GGML_METAL_HAS_TENSOR branch).
// Tile geometry matches ggml's proven params:
//   SZ_SIMDGROUP=16, N_MM_NK=2 -> NK=32, BLOCK_X=4, BLOCK_Y=2, SG_X=2, SG_Y=2
//   => NRA(M-tile)=64, NRB(N-tile)=128, 4 simdgroups, 128 threads.
//
// A = weights (Q8_0), dequantized to threadgroup half tile.
// B = activations (f32), read directly from device memory via a tensor view.
// C = f32 output, laid out [n*M + m] (M = weight rows = ne0, contiguous inner),
//     matching both ggml and the existing simdgroup kernel_mul_mm output layout.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

#define QK8_0 32
typedef struct {
    half d;            // delta
    int8_t qs[QK8_0];  // quants
} block_q8_0;

// Tile geometry (mirrors ggml-metal-impl.h)
#define MM2D_SZ_SG     16
#define MM2D_NK_BLK    2
#define MM2D_NK        (MM2D_SZ_SG * MM2D_NK_BLK)   // 32: K step per iteration
#define MM2D_BLOCK_X   4
#define MM2D_BLOCK_Y   2
#define MM2D_SG_X      2
#define MM2D_SG_Y      2
#define MM2D_NRB       (MM2D_SZ_SG * MM2D_BLOCK_X * MM2D_SG_X)  // 128: N tile
#define MM2D_NRA       (MM2D_SZ_SG * MM2D_BLOCK_Y * MM2D_SG_Y)  // 64:  M tile
#define MM2D_NSG       (MM2D_SG_X * MM2D_SG_Y)                  // 4 simdgroups
#define MM2D_NTHREADS  (32 * MM2D_NSG)                          // 128 threads

// Each block_q8_0 holds 16*nl weights with nl = 2.
#define MM2D_NL 2

// Dequantize 16 weights from a Q8_0 block (sub-tile `il` in 0..1) into a 4x4.
// Identical math to quantized.metal's dequantize_q8_0.
static inline void dequantize_q8_0_local(device const block_q8_0 *xb, short il, thread half4x4 & reg) {
    device const int8_t * qs = (device const int8_t *)xb->qs;
    const half d = xb->d;
    #pragma unroll(16)
    for (short i = 0; i < 16; i++) {
        reg[i/4][i%4] = (half)(qs[i + 16*il] * d);
    }
}

kernel void kernel_mul_mm_q8_0_f32_mm2d(
        device const  uchar * src0,   // weights, Q8_0
        device const  uchar * src1,   // activations, f32
        device        float * dst,
        constant    int64_t & ne00,   // K
        constant    int64_t & ne02,
        constant   uint64_t & nb01,   // weight row stride (bytes)
        constant   uint64_t & nb02,
        constant   uint64_t & nb03,
        constant    int64_t & ne12,
        constant   uint64_t & nb10,
        constant   uint64_t & nb11,   // activation row stride (bytes)
        constant   uint64_t & nb12,
        constant   uint64_t & nb13,
        constant    int64_t & ne0,    // M = weight rows
        constant    int64_t & ne1,    // N = activation rows
        constant       uint & r2,
        constant       uint & r3,
        threadgroup   uchar * shmem [[threadgroup(0)]],
        uint3                 tgpig [[threadgroup_position_in_grid]],
        uint                  tiitg [[thread_index_in_threadgroup]],
        uint                  sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;

    const int K = (int) ne00;
    const int M = (int) ne0;
    const int N = (int) ne1;

    const int im  = tgpig.z;
    const int i12 = im % (int) ne12;
    const int i13 = im / (int) ne12;

    const uint64_t offset0 = (i12 / r2) * nb02 + (i13 / r3) * nb03;

    // Output tile origin: ra over M (weight rows), rb over N (activation rows).
    const int ra = tgpig.y * MM2D_NRA;
    const int rb = tgpig.x * MM2D_NRB;

    // Threadgroup memory holds the dequantized A tile only: NRA rows x NK cols of half.
    threadgroup half * sa = (threadgroup half *)(shmem);

    auto tA = tensor(sa, dextents<int32_t, 2>(MM2D_NK, MM2D_NRA));

    // B (activations) read directly from device memory as f32, shape (K, N).
    device float * ptrB = (device float *)(src1 + nb12 * i12 + nb13 * i13);
    const int strideB = (int)(nb11 / sizeof(float));
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, strideB}));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            MM2D_NRB, MM2D_NRA, MM2D_NK, false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<MM2D_NSG>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    constexpr int A_WORK_ITEMS = MM2D_NRA * MM2D_NK_BLK; // one work item per (row, 16-wide k chunk)

    for (int loop_k = 0; loop_k < K; loop_k += MM2D_NK) {
        // PHASE 1: dequantize A tile (Q8_0 weights) into threadgroup memory.
        for (int work = tiitg; work < A_WORK_ITEMS; work += MM2D_NTHREADS) {
            const int row     = work / MM2D_NK_BLK;
            const int k_chunk = work % MM2D_NK_BLK;
            const int k_pos   = loop_k + k_chunk * 16;
            const short k_base = (short)(k_chunk * 16);

            if (ra + row < M) {
                const int   block_idx = k_pos / (16 * MM2D_NL);
                const short il        = (short)((k_pos / 16) % MM2D_NL);

                device const block_q8_0 * row_ptr =
                    (device const block_q8_0 *)(src0 + nb01 * (ra + row) + offset0);

                half4x4 temp_a;
                dequantize_q8_0_local(row_ptr + block_idx, il, temp_a);

                #pragma unroll(16)
                for (short i = 0; i < 16; i++) {
                    sa[row * MM2D_NK + (k_base + i)] = (k_pos + i < K) ? temp_a[i/4][i%4] : (half)0;
                }
            } else {
                #pragma unroll(16)
                for (short i = 0; i < 16; i++) {
                    sa[row * MM2D_NK + (k_base + i)] = (half)0;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // PHASE 2: cooperative tensor matmul, accumulate into cT.
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(loop_k, rb);
        mm.run(mB, mA, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store the result tile. Layout: dst[im*N*M + n*M + m].
    device float * dstBatch = (device float *)dst + (uint64_t)im * (uint64_t)N * (uint64_t)M;
    auto tD = tensor(dstBatch, dextents<int32_t, 2>(M, N), array<int, 2>({1, M}));
    cT.store(tD.slice(ra, rb));
}
