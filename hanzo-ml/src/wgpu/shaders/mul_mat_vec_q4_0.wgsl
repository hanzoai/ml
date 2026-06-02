// Q4_0 matrix-vector product (decode / memory-bound path): y[n] = sum_k W[n,k]*x[k] with W stored
// in the *native GGML* Q4_0 block format read straight from a GPU buffer -- NO CPU dequant round
// trip. Each row is K/32 blocks; one GGML BlockQ4_0 = 18 bytes = { f16 d ; u8 qs[16] }. The 32
// weights of a block are 4-bit: low nibble of qs[j] -> weight j, high nibble -> weight j+16, each
// dequantized as (nibble - 8) * d (byte-exact with k_quants.rs BlockQ4_0::to_float). One invocation
// computes one output element. WGSL port of mul_mat_vec_q4_0.comp; params in a UNIFORM at binding 0.

struct Params {
    nout: u32,
    k: u32,    // multiple of 32
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       w: array<u32>;  // raw Q4_0 blocks, 18 B each
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
    let rowbase = n * nblocks * 18u; // byte offset of row n
    var acc: f32 = 0.0;
    for (var b: u32 = 0u; b < nblocks; b = b + 1u) {
        let bb = rowbase + b * 18u;      // byte offset of this block
        let d = rdscale(bb);
        let qbase = bb + 2u;             // qs[] starts after the 2-byte scale
        let xb = b * 32u;                // activation base for this block
        var bsum: f32 = 0.0;
        for (var j: u32 = 0u; j < 16u; j = j + 1u) {
            let q = rdbyte(qbase + j);
            let x0 = f32(i32(q & 0x0Fu) - 8);  // low nibble -> weight j
            let x1 = f32(i32(q >> 4u) - 8);    // high nibble -> weight j+16
            bsum = bsum + x0 * x[xb + j];
            bsum = bsum + x1 * x[xb + j + 16u];
        }
        acc = acc + d * bsum;
    }
    y[n] = acc;
}
