// Batched naive matmul: C[bt] = A[bt](m x k) * B[bt](k x n), row-major, contiguous.
// Correctness-first port of the Vulkan bmm path. One invocation per output cell (bt,row,col).
// Params live in a UNIFORM buffer at binding 0 (portable across wgpu backends); the three storage
// buffers follow at bindings 1..3. Push order matches the Vulkan matmul dispatch: {batch,m,k,n}.

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
        let bo = bt * p.k * p.n;
        let co = bt * p.m * p.n;
        var acc: f32 = 0.0;
        for (var i: u32 = 0u; i < p.k; i = i + 1u) {
            acc = acc + a[ao + row * p.k + i] * b[bo + i * p.n + col];
        }
        c[co + row * p.n + col] = acc;
    }
}
