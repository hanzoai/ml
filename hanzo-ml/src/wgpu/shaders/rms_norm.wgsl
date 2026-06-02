// RMS norm over the last dim: y = x / sqrt(mean(x^2) + eps) * alpha. One invocation per row.
// Matches hanzo-ml CPU rms-norm: m = sqrt(sum(x^2)/dim + eps); y = x / m * alpha. WGSL port of rms_norm.comp.
struct Params { nrows: u32, m: u32, eps: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:     array<f32>;
@group(0) @binding(2) var<storage, read>       alpha: array<f32>;
@group(0) @binding(3) var<storage, read_write> y:     array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= p.nrows) { return; }
    let base = row * p.m;
    var ss: f32 = 0.0;
    for (var i: u32 = 0u; i < p.m; i = i + 1u) { let v = x[base + i]; ss = ss + v * v; }
    let denom = sqrt(ss / f32(p.m) + p.eps);
    for (var i: u32 = 0u; i < p.m; i = i + 1u) { y[base + i] = x[base + i] / denom * alpha[i]; }
}
