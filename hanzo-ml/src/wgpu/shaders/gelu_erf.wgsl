// Exact GELU: out = 0.5 * x * (1 + erf(x / sqrt(2))). (hanzo-ml "gelu_erf") WGSL port of gelu_erf.comp.
struct Params { n: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> o:   array<f32>;
fn erf_approx(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.3275911 * abs(x));
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                  - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
    return sign(x) * y;
}
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let x = inp[i];
    o[i] = 0.5 * x * (1.0 + erf_approx(x * 0.7071067811865476));
}
