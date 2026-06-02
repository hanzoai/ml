// Unary GELU (tanh approximation, matching hanzo-ml's CPU Gelu):
//   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))). WGSL port of gelu.comp.
struct Params { n: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> o:   array<f32>;
const SQRT_TWO_OVER_PI: f32 = 0.7978845608028654;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let x = inp[i];
    let inner = SQRT_TWO_OVER_PI * (x + 0.044715 * x * x * x);
    o[i] = 0.5 * x * (1.0 + tanh(inner));
}
