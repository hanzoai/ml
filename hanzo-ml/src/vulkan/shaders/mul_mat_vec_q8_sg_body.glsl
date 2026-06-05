// Subgroup-reduced Q8_0 matrix-vector product body: y[n] = sum_k W[n,k]*x[k], W Q8_0 (9 u32/block).
// Included by mul_mat_vec_q8_sg*.comp wrappers that #define CFG_RPS (rows/subgroup) before including;
// default 4. Decode is memory-bandwidth bound; one subgroup streams CFG_RPS output rows, loading each
// 32-block of x ONCE and reusing it across all rows -> arithmetic intensity + CFG_RPS independent
// weight loads in flight per lane (memory-level parallelism). Host dispatches ceil(nout/(subgroups*RPS)).
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_control_flow_attributes : enable

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer W { uint   w[]; };  // quantized weights, 9 u32 / 32-block
layout(set = 0, binding = 1) readonly buffer X { vec4   xv[]; }; // activation vector (vec4 view), length k/4
layout(set = 0, binding = 2) writeonly buffer Y { float y[]; };  // output, length nout
layout(push_constant) uniform Pc { uint nout; uint k; uint woff; };

#ifndef CFG_RPS
#define CFG_RPS 4u
#endif
const uint ROWS_PER_SG = CFG_RPS;

// Dequantized dot of one 32-block (vec4 activation loads; xb is a multiple of 32 -> xb/4 vec4-aligned).
float blockdot(uint off, uint xb) {
    float s = 0.0;
    uint xv0 = xb >> 2u;
    for (uint j = 0u; j < 8u; j++) {
        uint word = w[off + 1u + j];
        vec4 q = vec4(
            float(bitfieldExtract(int(word), 0, 8)),
            float(bitfieldExtract(int(word), 8, 8)),
            float(bitfieldExtract(int(word), 16, 8)),
            float(bitfieldExtract(int(word), 24, 8)));
        s += dot(q, xv[xv0 + j]);
    }
    return s;
}

void main() {
    uint sg = gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID;
    uint row0 = sg * ROWS_PER_SG;
    if (row0 >= nout) {
        return;
    }
    uint nblocks = k / 32u;
    uint lane = gl_SubgroupInvocationID;
    uint lanes = gl_SubgroupSize;

    // CFG_RPS rows streamed together; row0 is subgroup-uniform so hit[] is uniform (subgroupAdd safe).
    float acc[ROWS_PER_SG];
    uint  base[ROWS_PER_SG];
    bool  hit[ROWS_PER_SG];
    [[unroll]] for (uint r = 0u; r < ROWS_PER_SG; r++) {
        uint row = row0 + r;
        hit[r] = row < nout;
        base[r] = woff + row * nblocks * 9u;
        acc[r] = 0.0;
    }
    for (uint blk = lane; blk < nblocks; blk += lanes) {
        uint xb = blk * 32u;
        uint d = blk * 9u;
        [[unroll]] for (uint r = 0u; r < ROWS_PER_SG; r++)
            if (hit[r]) acc[r] += unpackHalf2x16(w[base[r] + d]).x * blockdot(base[r] + d, xb);
    }
    [[unroll]] for (uint r = 0u; r < ROWS_PER_SG; r++) {
        float t = subgroupAdd(acc[r]);
        if (subgroupElect() && hit[r]) y[row0 + r] = t;
    }
}
