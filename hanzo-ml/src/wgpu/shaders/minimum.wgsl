// Elementwise f32 minimum: out[i]=min(a[i],b[i]). WGSL port of minimum.comp.
struct Params { n: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       a: array<f32>;
@group(0) @binding(2) var<storage, read>       b: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < p.n) { o[i] = min(a[i], b[i]); }
}
