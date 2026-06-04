// Elementwise compare -> u32 {0,1}. op: 0=Eq 1=Ne 2=Le 3=Ge 4=Lt 5=Gt (hanzo-ml CmpOp order).
// WGSL port of cmp.comp.
struct Params { n: u32, op: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       a: array<f32>;
@group(0) @binding(2) var<storage, read>       b: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let x = a[i];
    let y = b[i];
    var r: bool;
    switch p.op {
        case 0u: { r = x == y; }
        case 1u: { r = x != y; }
        case 2u: { r = x <= y; }
        case 3u: { r = x >= y; }
        case 4u: { r = x <  y; }
        default: { r = x >  y; }
    }
    if (r) { o[i] = 1u; } else { o[i] = 0u; }
}
