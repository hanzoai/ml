// scatter add along `dim`. Same indexing as scatter_set but dst[idx] += src[g], done with an
// atomicCompareExchangeWeak loop on an atomic<u32> view of dst (WGSL has no float atomics).
// WGSL port of scatter_add_set.comp.
struct Params { n: u32, right: u32, dim_src: u32, dim_dst: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read>       src: array<f32>;
@group(0) @binding(3) var<storage, read>       ids: array<u32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    if (g >= p.n) { return; }
    let inner = g % p.right;
    let outer = g / (p.right * p.dim_src);
    let id = ids[g];
    let idx = outer * (p.dim_dst * p.right) + id * p.right + inner;
    let add = src[g];
    var old: u32 = atomicLoad(&dst[idx]);
    loop {
        let nv = bitcast<u32>(bitcast<f32>(old) + add);
        let res = atomicCompareExchangeWeak(&dst[idx], old, nv);
        if (res.exchanged) { break; }
        old = res.old_value;
    }
}
