// Affine transform: out[i] = mul * in[i] + add. WGSL port of affine.comp; params in a UNIFORM.
struct Params { n: u32, mul: f32, add: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> o:   array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < p.n) { o[i] = p.mul * inp[i] + p.add; }
}
