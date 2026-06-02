// Q8_0 matrix-vector product reading the *native GGML* Q8_0 block format straight from a GPU buffer
// (no CPU dequant, no re-pack). One GGML BlockQ8_0 = 34 bytes = { f16 d ; i8 qs[32] }; each weight
// dequantizes as qs[i] * d (byte-exact with k_quants.rs BlockQ8_0::to_float). 34 B/block is not
// 4-aligned, so the row is byte-addressed out of a u32-packed buffer. One invocation = one output
// element. WGSL port of mul_mat_vec_q8_0.comp; params in a UNIFORM buffer at binding 0.

struct Params {
    nout: u32,
    k: u32,    // multiple of 32
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       w: array<u32>;  // raw Q8_0 blocks, 34 B each
@group(0) @binding(2) var<storage, read>       x: array<f32>;  // activation vector, length k
@group(0) @binding(3) var<storage, read_write> y: array<f32>;  // output, length nout

// Read one byte at absolute byte-offset `bo` from the u32-packed weight buffer.
fn rdbyte(bo: u32) -> u32 {
    return extractBits(w[bo >> 2u], (bo & 3u) * 8u, 8u);
}
// Read the f16 block scale `d` (2 little-endian bytes at `bo`) as f32.
fn rdscale(bo: u32) -> f32 {
    let lo = rdbyte(bo);
    let hi = rdbyte(bo + 1u);
    return unpack2x16float(lo | (hi << 8u)).x;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= p.nout) {
        return;
    }
    let nblocks = p.k / 32u;
    let rowbase = n * nblocks * 34u; // byte offset of row n
    var acc: f32 = 0.0;
    for (var b: u32 = 0u; b < nblocks; b = b + 1u) {
        let bb = rowbase + b * 34u;
        let d = rdscale(bb);
        let qbase = bb + 2u;        // i8 qs[32] after the 2-byte scale
        let xb = b * 32u;
        var bsum: f32 = 0.0;
        for (var j: u32 = 0u; j < 32u; j = j + 1u) {
            // sign-extend the 8-bit weight lane.
            let q = extractBits(i32(rdbyte(qbase + j)), 0u, 8u);
            bsum = bsum + f32(q) * x[xb + j];
        }
        acc = acc + d * bsum;
    }
    y[n] = acc;
}
