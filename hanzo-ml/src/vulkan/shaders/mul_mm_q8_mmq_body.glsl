// RDNA3-style warp-tiled int8-dp4a Q8_0 GEMM body (the prefill lever, v2):
//   C[m,n] = A[m,k] * W[n,k]^T
// Ported from llama.cpp's ggml-vulkan mul_mmq.comp RDNA3 MMQ-int tiling onto OUR data layout
// (Q8_0 weights 9 u32/block w/ f16 scale at word 0; pre-quantized int8 activations 8 u32/block +
// f32 per-block scale) and OUR int8 dot (SPV OpSDotAccSat / sdot_accsat, FMT4x8). Keeps the Q8 weight
// read-once-per-workgroup shared staging, but replaces the flat 16x16 thread-grid with WARP-LEVEL
// tiling: BLOCK_SIZE threads split into BLOCK_SIZE/WARP "warps"; each warp owns a WM x WN region of the
// BM x BN output tile via WMITER x WNITER strided sub-tiles of TM x TN registers. The B operand for a
// column is pulled into a register (cache_b) ONCE and reused across all WMITER*TM A rows (register
// blocking), so shared-memory traffic in the inner loop drops from O(TM*TN) reads to O(TM+TN) per K.
// This is the occupancy+reuse structure llama uses for ~40% of peak; our flat grid was ~5%.
// CFG_DBUF=1 adds double-buffering (ping-pong shared buffers): the next K-step's global loads stream
// into the other buffer while dp4a consumes the current one, hiding memory latency behind compute
// (the latency-hider our flat dp4a GEMM had and the single-buffer port lacked).
// Config #defined by the wrapper before include: CFG_BM/CFG_BN (output tile, both % 16 handled via
// bound-checked store), CFG_BK_STEP (Q8 blocks staged per outer K iter), CFG_WM/CFG_WN (per-warp
// region), CFG_WMITER, CFG_TM/CFG_TN (register sub-tile), CFG_WARP (partition width; a tile constant,
// NOT a hw subgroup op -- no subgroup intrinsics are used, so correctness is layout-only),
// CFG_BLOCK_SIZE (workgroup threads), CFG_DBUF (0/1). BK is fixed 32 (one Q8 block). k % 32 == 0.
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_spirv_intrinsics : require
spirv_instruction(extensions = ["SPV_KHR_integer_dot_product"], capabilities = [6018, 6019], id = 4453)
  int sdot_accsat(int a, int b, int acc, spirv_literal int fmt);

const uint BM = CFG_BM;
const uint BN = CFG_BN;
const uint BK = 32u;                 // one Q8_0 block
const uint BK_STEP = CFG_BK_STEP;    // Q8 blocks staged per outer K iteration
const uint WM = CFG_WM;
const uint WN = CFG_WN;
const uint WMITER = CFG_WMITER;
const uint TM = CFG_TM;
const uint TN = CFG_TN;
const uint WARP = CFG_WARP;
const uint BLOCK_SIZE = CFG_BLOCK_SIZE;
const uint NBUF = (CFG_DBUF != 0) ? 2u : 1u;
const int  FMT4x8 = 0;

// Derived (mirror mul_mmq.comp): WNITER fills the rest of the warp region; WSUBM/WSUBN are the
// stride between a warp's successive register sub-tiles.
const uint WNITER = (WM * WN) / (WARP * TM * TN * WMITER);
const uint WSUBM  = WM / WMITER;
const uint WSUBN  = WN / WNITER;
const uint WARPS_M = BM / WM;        // warps tiling the M extent of the output tile

layout(local_size_x = CFG_BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer W  { uint  w[];  };  // Q8_0 weights, 9 u32 / 32-block, [n,k]
layout(set = 0, binding = 1) readonly buffer XQ { uint  xq[]; };  // int8 acts, 8 u32 / 32-block, [m, k/32]
// Aligned uvec4 alias of the activation buffer: each Q8 act block is 8 contiguous u32 at a 8-aligned
// base, so it loads as 2 16-byte-aligned uvec4 (xq4[abase/4 + 0], +1) instead of 8 scalar loads.
layout(set = 0, binding = 1) readonly buffer XQ4 { uvec4 xq4[]; };
layout(set = 0, binding = 2) readonly buffer XS { float xs[]; };  // act block scales, [m, k/32]
layout(set = 0, binding = 3) writeonly buffer Y { float y[];  };  // output f32, row-major [m, n]
layout(push_constant) uniform Pc { uint m; uint k; uint n; uint woff; };

// One staged Q8 block (8 int32 codes + f32 scale) for A and B.
struct blk_a { int qs[8]; float d; };
struct blk_b { int qs[8]; float d; };

shared blk_a buf_a[NBUF * BM * BK_STEP];
shared blk_b buf_b[NBUF * BN * BK_STEP];

// gl_WorkGroupID.x indexes M-tiles, .y indexes N-tiles (no K-split).
uint g_nblocks;
uint g_row0, g_col0;

// Stage K-blocks [kb0, kb0+BK_STEP) of this workgroup's tile into shared buffer slot `bi`.
void stage(uint bi, uint kb0) {
    uint abuf0 = bi * BM * BK_STEP;
    uint bbuf0 = bi * BN * BK_STEP;
    for (uint i = gl_LocalInvocationID.x; i < BM * BK_STEP; i += BLOCK_SIZE) {
        uint rr  = i % BM;
        uint sub = i / BM;
        uint gr  = g_row0 + rr;
        uint kb  = kb0 + sub;
        bool ok  = (gr < m) && (kb < g_nblocks);
        uint abase4 = ((gr * g_nblocks + kb) * 8u) >> 2; // /4: index into the uvec4 view
        uvec4 a0 = ok ? xq4[abase4]      : uvec4(0u);    // codes 0..3
        uvec4 a1 = ok ? xq4[abase4 + 1u] : uvec4(0u);    // codes 4..7
        buf_a[abuf0 + i].qs[0] = int(a0.x); buf_a[abuf0 + i].qs[1] = int(a0.y);
        buf_a[abuf0 + i].qs[2] = int(a0.z); buf_a[abuf0 + i].qs[3] = int(a0.w);
        buf_a[abuf0 + i].qs[4] = int(a1.x); buf_a[abuf0 + i].qs[5] = int(a1.y);
        buf_a[abuf0 + i].qs[6] = int(a1.z); buf_a[abuf0 + i].qs[7] = int(a1.w);
        buf_a[abuf0 + i].d = ok ? xs[gr * g_nblocks + kb] : 0.0;
    }
    // Coalesced weight staging: map consecutive threads to consecutive WORDS of a column-block.
    // Our Q8_0 weight block is 9 contiguous u32 (word 0 = f16 scale, 1..8 = int8 codes). Iterating
    // over (block, word) pairs with word = i%9 makes consecutive threads read consecutive global
    // addresses wbase+word, so a 9-long thread run hits one cache line instead of 9 threads each
    // stride-jumping nblocks*9 words apart (the dominant prefill bottleneck for our layout).
    uint nwords = BN * BK_STEP * 9u;
    for (uint i = gl_LocalInvocationID.x; i < nwords; i += BLOCK_SIZE) {
        uint item = i / 9u;            // which (cc, sub) column-block
        uint word = i - item * 9u;     // 0 = scale, 1..8 = code lane
        uint cc   = item % BN;
        uint sub  = item / BN;
        uint gc   = g_col0 + cc;
        uint kb   = kb0 + sub;
        bool ok   = (gc < n) && (kb < g_nblocks);
        uint wbase = woff + gc * g_nblocks * 9u + kb * 9u;
        uint raw  = ok ? w[wbase + word] : 0u;
        if (word == 0u) buf_b[bbuf0 + item].d = ok ? float(unpackHalf2x16(raw).x) : 0.0;
        else            buf_b[bbuf0 + item].qs[word - 1u] = int(raw);
    }
}

void main() {
    g_nblocks = k / 32u;
    g_row0 = gl_WorkGroupID.x * BM;
    g_col0 = gl_WorkGroupID.y * BN;

    // Warp / thread partition within the workgroup.
    uint warp_i = gl_LocalInvocationID.x / WARP;
    uint tiw    = gl_LocalInvocationID.x % WARP;
    uint tiwr   = tiw % (WSUBM / TM);      // thread row within a warp sub-tile
    uint tiwc   = tiw / (WSUBM / TM);      // thread col within a warp sub-tile
    uint warp_r = warp_i % WARPS_M;        // this warp's row in the BMxBN tile
    uint warp_c = warp_i / WARPS_M;        // this warp's col in the BMxBN tile

    // Per-thread accumulators: WMITER*WNITER sub-tiles of TM x TN.
    float sums[WMITER * TM * WNITER * TN];
    [[unroll]] for (uint i = 0u; i < WMITER * TM * WNITER * TN; i++) sums[i] = 0.0;

    // Register caches: the A rows this thread needs (reused across N), and one B column at a time.
    int   ca_qs[WMITER * TM][8];
    float ca_d [WMITER * TM];
    int   cb_qs[8];
    float cb_d;

    uint ksteps = (g_nblocks + BK_STEP - 1u) / BK_STEP;

    stage(0u, 0u);                          // prologue: first tile into slot 0
    barrier();

    for (uint kstep = 0u; kstep < ksteps; kstep++) {
        uint cur = (NBUF == 2u) ? (kstep & 1u) : 0u;
        if (NBUF == 2u && kstep + 1u < ksteps) {
            stage((kstep + 1u) & 1u, (kstep + 1u) * BK_STEP);   // prefetch next into the other slot
        }
        uint abuf0 = cur * BM * BK_STEP;
        uint bbuf0 = cur * BN * BK_STEP;

        // Consume the BK_STEP staged blocks of the current slot.
        [[unroll]] for (uint ks = 0u; ks < BK_STEP; ks++) {
            // Load this thread's A rows from shared into registers (reused across all N cols).
            [[unroll]] for (uint wsir = 0u; wsir < WMITER; wsir++) {
                [[unroll]] for (uint cr = 0u; cr < TM; cr++) {
                    uint reg_ib = wsir * TM + cr;
                    uint buf_ib = abuf0 + ks * BM + warp_r * WM + wsir * WSUBM + tiwr * TM + cr;
                    [[unroll]] for (uint q = 0u; q < 8u; q++) ca_qs[reg_ib][q] = buf_a[buf_ib].qs[q];
                    ca_d[reg_ib] = buf_a[buf_ib].d;
                }
            }
            // For each B column: pull it into a register ONCE, dp4a against all A rows.
            [[unroll]] for (uint wsic = 0u; wsic < WNITER; wsic++) {
                [[unroll]] for (uint cc = 0u; cc < TN; cc++) {
                    uint buf_ib = bbuf0 + ks * BN + warp_c * WN + wsic * WSUBN + tiwc * TN + cc;
                    [[unroll]] for (uint q = 0u; q < 8u; q++) cb_qs[q] = buf_b[buf_ib].qs[q];
                    cb_d = buf_b[buf_ib].d;

                    [[unroll]] for (uint wsir = 0u; wsir < WMITER; wsir++) {
                        [[unroll]] for (uint cr = 0u; cr < TM; cr++) {
                            uint reg_ib = wsir * TM + cr;
                            int dot = 0;
                            [[unroll]] for (uint q = 0u; q < 8u; q++)
                                dot = sdot_accsat(cb_qs[q], ca_qs[reg_ib][q], dot, FMT4x8);
                            uint si = (wsic * TN + cc) * (WMITER * TM) + reg_ib;
                            sums[si] += cb_d * ca_d[reg_ib] * float(dot);
                        }
                    }
                }
            }
        }

        // Single-buffer: reload the SAME slot for the next K-step after all reads are done.
        // Double-buffer: the prefetch above already filled the other slot; just sync.
        if (NBUF == 1u) {
            barrier();
            if (kstep + 1u < ksteps) stage(0u, (kstep + 1u) * BK_STEP);
        }
        barrier();
    }

    // ---- store: each thread writes its WMITER*WNITER sub-tiles of TM x TN ----
    uint dr = g_row0 + warp_r * WM;
    uint dc = g_col0 + warp_c * WN;
    [[unroll]] for (uint wsic = 0u; wsic < WNITER; wsic++) {
        [[unroll]] for (uint wsir = 0u; wsir < WMITER; wsir++) {
            uint dr_w = dr + wsir * WSUBM + tiwr * TM;
            uint dc_w = dc + wsic * WSUBN + tiwc * TN;
            [[unroll]] for (uint cc = 0u; cc < TN; cc++) {
                [[unroll]] for (uint cr = 0u; cr < TM; cr++) {
                    uint gr = dr_w + cr;
                    uint gc = dc_w + cc;
                    if (gr < m && gc < n) {
                        uint si = (wsic * TN + cc) * (WMITER * TM) + wsir * TM + cr;
                        y[gr * n + gc] = sums[si];
                    }
                }
            }
        }
    }
}
