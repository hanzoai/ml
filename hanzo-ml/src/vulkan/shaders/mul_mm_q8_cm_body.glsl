// Multi-warp coopmat (WMMA) Q8_0 GEMM body: C[m,n] = A[m,k](f32) * W[n,k](Q8_0)^T.
// Included by mul_mm_q8_cm*.comp wrappers that #define CFG_RM/CFG_RN before including (default 4x2).
// The fix over the single-subgroup coopmat (which lost to dp4a at 1/4 the occupancy): NSG=4 subgroups
// per 256-thread workgroup, each computing an RM x RN grid of 16x16 accumulator fragments for its own
// 32-col band, all sharing one cooperatively-staged tile. Per K-step the workgroup dequantizes a Q8
// weight tile + casts an activation tile to f16 in shared (each weight read once, reused across rows
// and the matrix-core MulAdds), then every subgroup runs coopMatMulAdd from shared. k%32==0; edges
// zero-padded on load, bound-checked on store.
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_control_flow_attributes : enable

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly  buffer W { uint      w[]; };  // Q8_0 weights, 9 u32/32-block, [n,k]
layout(set = 0, binding = 1) readonly  buffer A { float     x[]; };  // activations f32, [m,k]
layout(set = 0, binding = 2) writeonly buffer C { float     c[]; };  // output f32, [m,n]
layout(push_constant) uniform Pc { uint m; uint k; uint n; uint woff; };

#ifndef CFG_RM
#define CFG_RM 4u
#endif
#ifndef CFG_RN
#define CFG_RN 2u
#endif
const uint T   = 16u;
const uint RM  = CFG_RM;          // 16x16 acc tiles down M per subgroup
const uint RN  = CFG_RN;          // 16x16 acc tiles across N per subgroup
const uint NSG = 4u;              // subgroups per workgroup (256 threads / 64)
const uint BM  = RM * T;          // workgroup rows
const uint BNsg = RN * T;         // cols per subgroup
const uint BN  = NSG * BNsg;      // workgroup cols
const uint BK  = 32u;             // one Q8 block (2 coopmat K-substeps of 16)

shared float16_t sA[BM * BK];     // activation tile (shared by all subgroups)
shared float16_t sB[BN * BK];     // dequantized weight tile (each subgroup reads its band)

void main() {
    uint tid  = gl_LocalInvocationID.x;
    uint sg   = gl_SubgroupID;                 // 0..NSG-1
    uint row0 = gl_WorkGroupID.y * BM;
    uint col0 = gl_WorkGroupID.x * BN;
    uint nblocks = k / 32u;

    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc[RM][RN];
    [[unroll]] for (uint i = 0u; i < RM; i++)
        [[unroll]] for (uint j = 0u; j < RN; j++)
            acc[i][j] = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);

    for (uint kb = 0u; kb < nblocks; kb++) {
        // activation tile -> sA (f16)
        for (uint i = tid; i < BM * BK; i += 256u) {
            uint r = i / BK, j = i % BK; uint gr = row0 + r;
            sA[i] = (gr < m) ? float16_t(x[gr * k + kb * BK + j]) : float16_t(0.0);
        }
        // weight tile: dequant Q8 -> sB (f16)
        for (uint i = tid; i < BN * BK; i += 256u) {
            uint cc = i / BK, j = i % BK; uint gc = col0 + cc;
            if (gc < n) {
                uint off = woff + gc * nblocks * 9u + kb * 9u;
                float scale = unpackHalf2x16(w[off]).x;
                int q = bitfieldExtract(int(w[off + 1u + j / 4u]), int((j % 4u) * 8u), 8);
                sB[i] = float16_t(scale * float(q));
            } else {
                sB[i] = float16_t(0.0);
            }
        }
        barrier();

        [[unroll]] for (uint ks = 0u; ks < BK; ks += T) {
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> ma[RM];
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> mb[RN];
            [[unroll]] for (uint i = 0u; i < RM; i++)
                coopMatLoad(ma[i], sA, i * T * BK + ks, BK, gl_CooperativeMatrixLayoutRowMajor);
            [[unroll]] for (uint j = 0u; j < RN; j++)
                coopMatLoad(mb[j], sB, (sg * BNsg + j * T) * BK + ks, BK, gl_CooperativeMatrixLayoutColumnMajor);
            [[unroll]] for (uint i = 0u; i < RM; i++)
                [[unroll]] for (uint j = 0u; j < RN; j++)
                    acc[i][j] = coopMatMulAdd(ma[i], mb[j], acc[i][j]);
        }
        barrier();
    }

    [[unroll]] for (uint i = 0u; i < RM; i++)
        [[unroll]] for (uint j = 0u; j < RN; j++) {
            uint r = row0 + i * T;
            uint cc = col0 + sg * BNsg + j * T;
            if (r < m && cc < n)
                coopMatStore(acc[i][j], c, r * n + cc, n, gl_CooperativeMatrixLayoutRowMajor);
        }
}
