// Fused MoE grouped quant matvec (Q4_0): y[s, r] = sum_k W[ids[s], r, k] * x[s, k], reading the
// per-expert slice from a GGML Q4_0 weight bank [E, n, k] in VRAM. Router gather + per-expert GEMM
// in one dispatch. Q4_0 block = 18 bytes = { f16 d ; u8 qs[16] }. One invocation = one y element.
// WGSL port of moe_matvec_q4_0.comp; params in a UNIFORM at binding 0.
struct Params { n: u32, k: u32, nrows: u32 };  // k mult of 32
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       w:   array<u32>;  // expert bank, raw Q4_0 bytes
@group(0) @binding(2) var<storage, read>       x:   array<f32>;  // [S, k] activations
@group(0) @binding(3) var<storage, read>       ids: array<u32>;  // [S] expert id per slot
@group(0) @binding(4) var<storage, read_write> y:   array<f32>;  // [S, n] outputs

fn rdbyte(bo: u32) -> u32 { return extractBits(w[bo >> 2u], (bo & 3u) * 8u, 8u); }
fn rdscale(bo: u32) -> f32 {
    let lo = rdbyte(bo);
    let hi = rdbyte(bo + 1u);
    return unpack2x16float(lo | (hi << 8u)).x;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    let total = p.nrows * p.n;
    if (g >= total) { return; }
    let s = g / p.n;
    let r = g - s * p.n;
    let expert = ids[s];
    let nblocks = p.k / 32u;
    let rowbase = (expert * p.n + r) * nblocks * 18u;
    let xbase = s * p.k;
    var acc: f32 = 0.0;
    for (var b: u32 = 0u; b < nblocks; b = b + 1u) {
        let bb = rowbase + b * 18u;
        let d = rdscale(bb);
        let qbase = bb + 2u;
        let xb = xbase + b * 32u;
        var bsum: f32 = 0.0;
        for (var j: u32 = 0u; j < 16u; j = j + 1u) {
            let q = rdbyte(qbase + j);
            let x0 = f32(i32(q & 0x0Fu) - 8);
            let x1 = f32(i32(q >> 4u) - 8);
            bsum = bsum + x0 * x[xb + j];
            bsum = bsum + x1 * x[xb + j + 16u];
        }
        acc = acc + d * bsum;
    }
    y[g] = acc;
}
