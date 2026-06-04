// to_dtype F32 -> U32: out[i] = uint(max(in[i],0)) (truncates toward zero). WGSL port of cast_f2u.comp.
struct Params { n: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> o:   array<u32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < p.n) { o[i] = u32(max(inp[i], 0.0)); }
}
