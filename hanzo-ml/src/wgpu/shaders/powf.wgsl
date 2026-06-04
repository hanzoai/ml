// Unary powf with a scalar exponent: out[i] = pow(in[i], e). WGSL port of powf.comp.
struct Params { n: u32, e: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> o:   array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    o[i] = pow(inp[i], p.e);
}
