// Q4_K matrix-vector product reading the *native GGML* Q4_K super-block format straight from a GPU
// buffer (no CPU dequant). One GGML BlockQ4K = 144 bytes = { f16 d ; f16 dmin ; u8 scales[12] ;
// u8 qs[128] } holding 256 weights in 8 sub-blocks of 32. Dequant byte-exact with k_quants.rs
// BlockQ4K::to_float: weight = d*sc*nibble - dmin*m, (sc,m) from get_scale_min_k4. One invocation =
// one output row. WGSL port of mul_mat_vec_q4k.comp; params in a UNIFORM at binding 0.
struct Params { nout: u32, k: u32 };  // k a multiple of 256
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       w: array<u32>;  // raw Q4_K blocks, 144 B each
@group(0) @binding(2) var<storage, read>       x: array<f32>;  // activation vector, length k
@group(0) @binding(3) var<storage, read_write> y: array<f32>;  // output, length nout

fn rdbyte(bo: u32) -> u32 {
    return extractBits(w[bo >> 2u], (bo & 3u) * 8u, 8u);
}
fn rdscale(bo: u32) -> f32 {
    let lo = rdbyte(bo);
    let hi = rdbyte(bo + 1u);
    return unpack2x16float(lo | (hi << 8u)).x;
}
// get_scale_min_k4(j, scales) -> (scale6bit, min6bit) read from the 12 packed scale bytes at sbase.
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
    let n = gid.x;
    if (n >= p.nout) { return; }
    let nsb = p.k / 256u;
    let rowbase = n * nsb * 144u;
    var acc: f32 = 0.0;
    for (var sb: u32 = 0u; sb < nsb; sb = sb + 1u) {
        let bb = rowbase + sb * 144u;
        let d    = rdscale(bb);
        let dmin = rdscale(bb + 2u);
        let sbase = bb + 4u;
        let qbase = bb + 16u;
        let xsb = sb * 256u;
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
                let lo = f32(q & 0x0Fu);
                let hi = f32(q >> 4u);
                acc = acc + (d1 * lo - mm1) * x[xsb + joff + l];
                acc = acc + (d2 * hi - mm2) * x[xsb + joff + 32u + l];
            }
            is = is + 2u;
        }
    }
    y[n] = acc;
}
