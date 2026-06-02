// gather along `dim`. ids has the OUTPUT shape; for output flat index g decomposed as
// (outer, j, inner) with inner < right and j < dim_out, the source element is
//   src[outer*(dim_src*right) + ids[g]*right + inner]. WGSL port of gather.comp.
struct Params { n: u32, right: u32, dim_out: u32, dim_src: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       src: array<f32>;
@group(0) @binding(2) var<storage, read>       ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> o:   array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    if (g >= p.n) { return; }
    let inner = g % p.right;
    let outer = g / (p.right * p.dim_out);
    let id = ids[g];
    o[g] = src[outer * (p.dim_src * p.right) + id * p.right + inner];
}
