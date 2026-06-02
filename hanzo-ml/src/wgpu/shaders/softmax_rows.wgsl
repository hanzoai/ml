// Numerically-stable softmax over the last dim. One invocation per row (row = all but the last
// dim flattened). WGSL port of softmax_rows.comp.
struct Params { nrows: u32, m: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= p.nrows) { return; }
    let base = row * p.m;
    var mx: f32 = -3.402823466e38;
    for (var i: u32 = 0u; i < p.m; i = i + 1u) { mx = max(mx, x[base + i]); }
    var s: f32 = 0.0;
    for (var i: u32 = 0u; i < p.m; i = i + 1u) {
        let e = exp(x[base + i] - mx);
        y[base + i] = e;
        s = s + e;
    }
    let inv = 1.0 / s;
    for (var i: u32 = 0u; i < p.m; i = i + 1u) { y[base + i] = y[base + i] * inv; }
}
