// Shared-memory-tiled, DOUBLE-BUFFERED int8-dot (dp4a) Q8_0 GEMM body (the prefill lever):
//   C[m,n] = A[m,k] * W[n,k]^T
// Included by mul_mm_q8_dp4a*.comp wrappers that #define the tile config (CFG_TM/CFG_TN/CFG_NBK)
// before including; defaults below = the committed 64x64/NBK=2. 256 threads always (BM/TM==BN/TN==16).
// A BM x BN output tile per workgroup; 256 threads each own a TM x TN register sub-tile. Each K-step
// stages NBK Q8 blocks of weights+int8-acts into shared so every Q8 weight is read ONCE per workgroup
// and reused across all BM rows (weight reuse AND occupancy together). Two shared buffers are
// ping-ponged: while dp4a consumes tile k, the global loads for tile k+1 stream into the other buffer,
// hiding memory latency behind compute. Acts pre-quantized int8 (xq) + per-block f32 scale (xs);
// weights Q8_0 (9 u32/block). k % 32 == 0; m,n arbitrary (edges/tail zero-padded, bound-checked store).
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_spirv_intrinsics : require
spirv_instruction(extensions = ["SPV_KHR_integer_dot_product"], capabilities = [6018, 6019], id = 4453)
  int sdot_accsat(int a, int b, int acc, spirv_literal int fmt);

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer W  { uint  w[];  };  // Q8_0 weights, 9 u32 / 32-block, [n,k]
layout(set = 0, binding = 1) readonly buffer XQ { uint  xq[]; };  // int8 acts, 4/u32, [m, k/32, 8]
layout(set = 0, binding = 2) readonly buffer XS { float xs[]; };  // act block scales, [m, k/32]
layout(set = 0, binding = 3) writeonly buffer Y { float y[]; };   // output f32, row-major [m, n]
layout(push_constant) uniform Pc { uint m; uint k; uint n; uint woff; };

#ifndef CFG_TM
#define CFG_TM 4u
#endif
#ifndef CFG_TN
#define CFG_TN 4u
#endif
#ifndef CFG_NBK
#define CFG_NBK 2u
#endif
const uint TM  = CFG_TM, TN = CFG_TN; // per-thread register sub-tile
const uint BM  = 16u * TM, BN = 16u * TN; // (BM/TM)*(BN/TN) = 16*16 = 256 threads
const uint NBK = CFG_NBK;              // Q8 blocks staged per K-step
const uint BKU = NBK * 8u;             // u32 per row staged in shared
const int  FMT4x8 = 0;

shared uint  sXq[2][BM * BKU];        // double-buffered int8 activations
shared uint  sWq[2][BN * BKU];        // double-buffered Q8 weight payload
shared float sXs[2][BM * NBK];        // activation block scales
shared float sWs[2][BN * NBK];        // weight block scales

// Stage K-step `kb0`'s tile (NBK blocks) into shared buffer `bi`. Reads gl_ builtins directly.
void load_tile(uint bi, uint kb0) {
    uint tid  = gl_LocalInvocationID.x;
    uint row0 = gl_WorkGroupID.y * BM;
    uint col0 = gl_WorkGroupID.x * BN;
    uint nblocks = k / 32u;
    for (uint i = tid; i < BM * BKU; i += 256u) {
        uint rr = i / BKU, idx = i % BKU, sub = idx / 8u, ww = idx % 8u; uint gr = row0 + rr;
        sXq[bi][i] = (gr < m && kb0 + sub < nblocks) ? xq[(gr * nblocks + kb0 + sub) * 8u + ww] : 0u;
    }
    for (uint i = tid; i < BM * NBK; i += 256u) {
        uint rr = i / NBK, sub = i % NBK; uint gr = row0 + rr;
        sXs[bi][i] = (gr < m && kb0 + sub < nblocks) ? xs[gr * nblocks + kb0 + sub] : 0.0;
    }
    for (uint i = tid; i < BN * BKU; i += 256u) {
        uint cc = i / BKU, idx = i % BKU, sub = idx / 8u, ww = idx % 8u; uint gc = col0 + cc;
        sWq[bi][i] = (gc < n && kb0 + sub < nblocks)
            ? w[woff + gc * nblocks * 9u + (kb0 + sub) * 9u + 1u + ww] : 0u;
    }
    for (uint i = tid; i < BN * NBK; i += 256u) {
        uint cc = i / NBK, sub = i % NBK; uint gc = col0 + cc;
        sWs[bi][i] = (gc < n && kb0 + sub < nblocks)
            ? unpackHalf2x16(w[woff + gc * nblocks * 9u + (kb0 + sub) * 9u]).x : 0.0;
    }
}

void main() {
    uint tid  = gl_LocalInvocationID.x;        // 0..255
    uint tr   = tid / 16u;                     // thread-row 0..15
    uint tc   = tid % 16u;                     // thread-col 0..15
    uint row0 = gl_WorkGroupID.y * BM;
    uint col0 = gl_WorkGroupID.x * BN;
    uint nblocks = k / 32u;
    uint ksteps  = (nblocks + NBK - 1u) / NBK;

    float acc[TM][TN];
    [[unroll]] for (uint i = 0u; i < TM; i++)
        [[unroll]] for (uint j = 0u; j < TN; j++) acc[i][j] = 0.0;

    load_tile(0u, 0u);                         // prologue: first tile into buffer 0
    barrier();

    for (uint kstep = 0u; kstep < ksteps; kstep++) {
        uint cur = kstep & 1u;
        if (kstep + 1u < ksteps) load_tile((kstep + 1u) & 1u, (kstep + 1u) * NBK); // prefetch next
        [[unroll]] for (uint ti = 0u; ti < TM; ti++) {
            uint rr = tr * TM + ti;
            [[unroll]] for (uint tj = 0u; tj < TN; tj++) {
                uint cc = tc * TN + tj;
                [[unroll]] for (uint sub = 0u; sub < NBK; sub++) {
                    int dot = 0;
                    [[unroll]] for (uint ww = 0u; ww < 8u; ww++)
                        dot = sdot_accsat(int(sWq[cur][cc * BKU + sub * 8u + ww]),
                                          int(sXq[cur][rr * BKU + sub * 8u + ww]), dot, FMT4x8);
                    acc[ti][tj] += sWs[cur][cc * NBK + sub] * sXs[cur][rr * NBK + sub] * float(dot);
                }
            }
        }
        barrier();
    }

    [[unroll]] for (uint ti = 0u; ti < TM; ti++) {
        uint gr = row0 + tr * TM + ti;
        [[unroll]] for (uint tj = 0u; tj < TN; tj++) {
            uint gc = col0 + tc * TN + tj;
            if (gr < m && gc < n) y[gr * n + gc] = acc[ti][tj];
        }
    }
}
