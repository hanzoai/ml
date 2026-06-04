// Row-wise argmin over the last dim: out[row] = index of the extremum in row. Output u32. WGSL port.
struct Params { rows: u32, cols: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> o:   array<u32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= p.rows) { return; }
    let base = row * p.cols;
    var best: f32 = inp[base];
    var bi: u32 = 0u;
    for (var c: u32 = 1u; c < p.cols; c = c + 1u) {
        let v = inp[base + c];
        if (v < best) { best = v; bi = c; }
    }
    o[row] = bi;
}
