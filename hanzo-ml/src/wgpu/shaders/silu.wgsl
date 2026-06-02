// Unary SiLU: out[i] = x/(1+exp(-x)). WGSL port of silu.comp; params in a UNIFORM at binding 0.
struct Params { n: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> o:   array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let x = inp[i];
    o[i] = x / (1.0 + exp(-x));
}
