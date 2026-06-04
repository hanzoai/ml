// scatter set along `dim`. ids/src share a shape; for src flat index g = (outer, j, inner)
// with inner < right and j < dim_src, write dst[outer*(dim_dst*right) + ids[g]*right + inner] = src[g].
// dst is binding 0 (the kernel writes it). WGSL port of scatter_set.comp.
struct Params { n: u32, right: u32, dim_src: u32, dim_dst: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read>       src: array<f32>;
@group(0) @binding(3) var<storage, read>       ids: array<u32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    if (g >= p.n) { return; }
    let inner = g % p.right;
    let outer = g / (p.right * p.dim_src);
    let id = ids[g];
    dst[outer * (p.dim_dst * p.right) + id * p.right + inner] = src[g];
}
