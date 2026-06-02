// Fill a contiguous run of `n` 4-byte words with a constant bit pattern `val`: out[i] = val for
// i in [0, n). Powers const_set / Tensor::full / ones on a contiguous, offset-0, whole-buffer view.
// `val` is a raw 32-bit pattern so the SAME kernel serves f32 and u32 storage bit-exactly. WGSL port
// of const_fill.comp; params in a UNIFORM at binding 0.

struct Params {
    n: u32,
    val: u32, // raw 32-bit pattern to store
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> outp: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < p.n) {
        outp[i] = p.val;
    }
}
