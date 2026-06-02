// Materialize a CONTIGUOUS f32 output from an arbitrarily-strided source. Powers .contiguous(),
// transpose, and broadcast. For each contiguous output index gid (< n), decode gid into a
// multi-index over the logical shape (row-major, last dim fastest), map through strides+offset
// (in ELEMENTS) into `inp`, and write inp[src] -> outp[gid+dst_offset]. stride 0 broadcasts. rank<=6.
// WGSL port of strided_copy.comp. shape/strides are 6 scalar fields (not a WGSL array) so the
// uniform layout is tightly predictable and the Rust side packs it byte-for-byte.

struct Params {
    n: u32,
    rank: u32,
    offset: u32,
    dst_offset: u32,
    s0: u32, s1: u32, s2: u32, s3: u32, s4: u32, s5: u32,        // shape[0..6]
    d0: u32, d1: u32, d2: u32, d3: u32, d4: u32, d5: u32,        // strides[0..6]
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       inp:  array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;

fn shape_at(d: u32) -> u32 {
    switch d {
        case 0u: { return p.s0; }
        case 1u: { return p.s1; }
        case 2u: { return p.s2; }
        case 3u: { return p.s3; }
        case 4u: { return p.s4; }
        default: { return p.s5; }
    }
}
fn stride_at(d: u32) -> u32 {
    switch d {
        case 0u: { return p.d0; }
        case 1u: { return p.d1; }
        case 2u: { return p.d2; }
        case 3u: { return p.d3; }
        case 4u: { return p.d4; }
        default: { return p.d5; }
    }
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    if (g < p.n) {
        var rem = g;
        var src = p.offset;
        // Peel the last dim first (row-major: last dim varies fastest).
        for (var d: u32 = 0u; d < p.rank; d = d + 1u) {
            let dd = p.rank - 1u - d;
            let dim = shape_at(dd);
            let idx = rem % dim;
            rem = rem / dim;
            src = src + idx * stride_at(dd);
        }
        outp[g + p.dst_offset] = inp[src];
    }
}
