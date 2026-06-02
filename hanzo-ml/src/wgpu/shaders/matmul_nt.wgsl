// Batched "NT" matmul: C[bt](m x n) = A[bt](m x k) * B[bt](n x k)^T, row-major, contiguous.
// B is stored row-major as [n, k] (a weight matrix W passed without an explicit transpose copy),
// so element (col, i) of B is b[bo + col*k + i]. This mirrors the Vulkan bmm_reg_nt path and lets
// a Linear feed W[n,k] directly, skipping the transpose materialization. Push order: {batch,m,k,n}.

struct Params {
    batch: u32,
    m: u32,
    k: u32,
    n: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       a: array<f32>;
@group(0) @binding(2) var<storage, read>       b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bt = gid.z;
    let row = gid.y;
    let col = gid.x;
    if (bt < p.batch && row < p.m && col < p.n) {
        let ao = bt * p.m * p.k;
        let bo = bt * p.n * p.k;
        let co = bt * p.m * p.n;
        var acc: f32 = 0.0;
        for (var i: u32 = 0u; i < p.k; i = i + 1u) {
            acc = acc + a[ao + row * p.k + i] * b[bo + col * p.k + i];
        }
        c[co + row * p.n + col] = acc;
    }
}
