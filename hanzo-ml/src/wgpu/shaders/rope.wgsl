// GPT-NeoX style rotary embedding (hanzo-ml "rotary-emb"). src is [b,h,t,d] contiguous;
// cos/sin are [t,d/2] (unbatched) or [b,t,d/2]. One invocation per (bh, t, d/2) triple.
//   dst[i1] = src[i1]*cos - src[i2]*sin ;  dst[i2] = src[i1]*sin + src[i2]*cos
// WGSL port of rope.comp.
struct Params { b: u32, h: u32, t: u32, d: u32, unbatched: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       src:  array<f32>;
@group(0) @binding(2) var<storage, read>       cosb: array<f32>;
@group(0) @binding(3) var<storage, read>       sinb: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst:  array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    let hd = p.d / 2u;
    let per_bh = p.t * hd;
    let total = p.b * p.h * per_bh;
    if (g >= total) { return; }
    let bh_i = g / per_bh;
    let rem = g % per_bh;
    let i_t = rem / hd;
    let i_d = rem % hd;
    let sbase = bh_i * p.t * p.d;
    let i1 = sbase + i_t * p.d + i_d;
    let i2 = i1 + hd;
    var i_cs = i_t * hd + i_d;
    if (p.unbatched != 0u) { i_cs = i_cs + (bh_i / p.h) * per_bh; }
    let c = cosb[i_cs];
    let s = sinb[i_cs];
    let x1 = src[i1];
    let x2 = src[i2];
    dst[i1] = x1 * c - x2 * s;
    dst[i2] = x1 * s + x2 * c;
}
