// Fused MoE grouped quant matvec (Q4_K): y[s, r] = sum_k W[ids[s], r, k] * x[s, k], reading the
// per-expert slice from a GGML Q4_K weight bank [E, n, k] in VRAM. Router gather + per-expert GEMM
// in one dispatch. Q4_K super-block = 144 bytes (256 weights, 8 sub-blocks of 32, 6-bit packed
// scales/mins); dequant = d*sc*nibble - dmin*m. One invocation = one y element. WGSL port of
// moe_matvec_q4k.comp; params in a UNIFORM at binding 0.
struct Params { n: u32, k: u32, nrows: u32 };  // k mult of 256
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       w:   array<u32>;  // expert bank, raw Q4_K bytes
@group(0) @binding(2) var<storage, read>       x:   array<f32>;  // [S, k] activations
@group(0) @binding(3) var<storage, read>       ids: array<u32>;  // [S] expert id per slot
@group(0) @binding(4) var<storage, read_write> y:   array<f32>;  // [S, n] outputs

fn rdbyte(bo: u32) -> u32 { return extractBits(w[bo >> 2u], (bo & 3u) * 8u, 8u); }
fn rdscale(bo: u32) -> f32 {
    let lo = rdbyte(bo);
    let hi = rdbyte(bo + 1u);
    return unpack2x16float(lo | (hi << 8u)).x;
}
struct ScaleMin { sc: u32, m: u32 };
fn scale_min(sbase: u32, j: u32) -> ScaleMin {
    var out: ScaleMin;
    if (j < 4u) {
        out.sc = rdbyte(sbase + j) & 63u;
        out.m  = rdbyte(sbase + j + 4u) & 63u;
    } else {
        let a = rdbyte(sbase + j + 4u);
        let b4 = rdbyte(sbase + j - 4u);
        let bj = rdbyte(sbase + j);
        out.sc = (a & 0x0Fu) | ((b4 >> 6u) << 4u);
        out.m  = (a >> 4u)   | ((bj >> 6u) << 4u);
    }
    return out;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    let total = p.nrows * p.n;
    if (g >= total) { return; }
    let s = g / p.n;
    let r = g - s * p.n;
    let expert = ids[s];
    let nsb = p.k / 256u;
    let rowbase = (expert * p.n + r) * nsb * 144u;
    let xbase = s * p.k;
    var acc: f32 = 0.0;
    for (var sb: u32 = 0u; sb < nsb; sb = sb + 1u) {
        let bb = rowbase + sb * 144u;
        let d    = rdscale(bb);
        let dmin = rdscale(bb + 2u);
        let sbase = bb + 4u;
        let qbase = bb + 16u;
        let xsb = xbase + sb * 256u;
        var is: u32 = 0u;
        for (var grp: u32 = 0u; grp < 4u; grp = grp + 1u) {
            let joff = grp * 64u;
            let qoff = qbase + grp * 32u;
            let sm1 = scale_min(sbase, is);
            let sm2 = scale_min(sbase, is + 1u);
            let d1 = d * f32(sm1.sc); let mm1 = dmin * f32(sm1.m);
            let d2 = d * f32(sm2.sc); let mm2 = dmin * f32(sm2.m);
            for (var l: u32 = 0u; l < 32u; l = l + 1u) {
                let q = rdbyte(qoff + l);
                acc = acc + (d1 * f32(q & 0x0Fu) - mm1) * x[xsb + joff + l];
                acc = acc + (d2 * f32(q >> 4u)   - mm2) * x[xsb + joff + 32u + l];
            }
            is = is + 2u;
        }
    }
    y[g] = acc;
}
