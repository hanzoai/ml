// Row-wise max reduction over the last dim of a row-major [rows, cols] f32 buffer. WGSL port.
struct Params { rows: u32, cols: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> o:   array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row < p.rows) {
        let base = row * p.cols;
        var acc: f32 = inp[base];
        for (var c: u32 = 1u; c < p.cols; c = c + 1u) { acc = max(acc, inp[base + c]); }
        o[row] = acc;
    }
}
