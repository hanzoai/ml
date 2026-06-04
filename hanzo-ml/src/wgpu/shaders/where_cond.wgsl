// Elementwise select: outp[i] = (cond[i] != 0) ? t[i] : f[i]. All buffers length n.
// WGSL port of where_cond.comp.
struct Params { n: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       cond: array<u32>;
@group(0) @binding(2) var<storage, read>       t:    array<f32>;
@group(0) @binding(3) var<storage, read>       f:    array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    if (g < p.n) {
        if (cond[g] != 0u) { outp[g] = t[g]; } else { outp[g] = f[g]; }
    }
}
