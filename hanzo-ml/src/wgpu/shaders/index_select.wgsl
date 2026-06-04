// Gather rows along a dim (embeddings). Source is viewed as [left, dim_size, right] row-major;
// `ids` selects n_ids indices along the middle dim, producing a [left, n_ids, right] f32 output.
// total = left*n_ids*right. WGSL port of index_select.comp.
struct Params { left: u32, dim_size: u32, right: u32, n_ids: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       ids:  array<u32>;
@group(0) @binding(2) var<storage, read>       inp:  array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    let total = p.left * p.n_ids * p.right;
    if (g < total) {
        let r = g % p.right;
        let i = (g / p.right) % p.n_ids;
        let l = g / (p.right * p.n_ids);
        outp[g] = inp[(l * p.dim_size + ids[i]) * p.right + r];
    }
}
