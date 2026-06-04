// 2D strided block copy: copy `d1` rows of `d2` contiguous 4-byte words from `inp` to `outp`, with
// independent per-row source/dest strides and base offsets (candle's copy2d: cat/slice_set, the
// KV-cache append + GQA repeat_kv path). Typed u32 so it is bit-exact for BOTH f32 and u32 storage.
// Output keeps any element this copy does not address (never zeroed) so successive writes compose.
// WGSL port of copy2d.comp; params in a UNIFORM at binding 0.

struct Params {
    d1: u32,          // number of rows
    d2: u32,          // contiguous elements per row
    src_stride1: u32, // elements between consecutive source rows
    dst_stride1: u32, // elements between consecutive dest rows
    src_offset: u32,  // base element offset into inp
    dst_offset: u32,  // base element offset into outp
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       inp:  array<u32>;
@group(0) @binding(2) var<storage, read_write> outp: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    let total = p.d1 * p.d2;
    if (g < total) {
        let row = g / p.d2;
        let col = g - row * p.d2;
        let s = p.src_offset + row * p.src_stride1 + col;
        let d = p.dst_offset + row * p.dst_stride1 + col;
        outp[d] = inp[s];
    }
}
