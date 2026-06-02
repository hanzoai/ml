// Fused scaled-dot-product attention (flash-attention style, online softmax) for one query row.
//   out[bh, qi, :] = sum_j softmax_j( scale * dot(Q[bh,qi], K[bh,j]) (+ causal mask) ) * V[bh, j, :]
// One invocation computes one full output row (head_dim D values), streaming over the Lk keys with
// the numerically-stable running-max / running-sum recurrence -- the [Lq, Lk] score matrix is never
// materialized. Q/K/V are contiguous [BH, L, D] f32. `causal` masks key j > (qi + key_len - q_len).
// WGSL port of flash_attn.comp; params in a UNIFORM at binding 0.
struct Params {
    bh: u32,
    lq: u32,
    lk: u32,
    d: u32,
    scale: f32,
    causal: u32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q: array<f32>;  // [BH, Lq, D]
@group(0) @binding(2) var<storage, read>       k: array<f32>;  // [BH, Lk, D]
@group(0) @binding(3) var<storage, read>       v: array<f32>;  // [BH, Lk, D]
@group(0) @binding(4) var<storage, read_write> o: array<f32>;  // [BH, Lq, D]

const DMAX: u32 = 256u; // supports head_dim up to 256 (Qwen3 uses 128)

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    let total = p.bh * p.lq;
    if (g >= total) { return; }
    let b = g / p.lq;
    let qi = g - b * p.lq;
    let qbase = (b * p.lq + qi) * p.d;
    let kv_slice = b * p.lk * p.d;

    var last_key = p.lk;
    if (p.causal != 0u) {
        last_key = qi + (p.lk - p.lq) + 1u;
        if (last_key > p.lk) { last_key = p.lk; }
    }

    var acc: array<f32, 256>;
    for (var t: u32 = 0u; t < p.d; t = t + 1u) { acc[t] = 0.0; }
    var m: f32 = -3.402823466e38;
    var l: f32 = 0.0;

    for (var j: u32 = 0u; j < last_key; j = j + 1u) {
        let kbase = kv_slice + j * p.d;
        var s: f32 = 0.0;
        for (var t: u32 = 0u; t < p.d; t = t + 1u) {
            s = s + q[qbase + t] * k[kbase + t];
        }
        s = s * p.scale;
        let mnew = max(m, s);
        let corr = exp(m - mnew);
        let pr = exp(s - mnew);
        let vbase = kv_slice + j * p.d;
        for (var t: u32 = 0u; t < p.d; t = t + 1u) {
            acc[t] = acc[t] * corr + pr * v[vbase + t];
        }
        l = l * corr + pr;
        m = mnew;
    }

    var inv: f32 = 0.0;
    if (l > 0.0) { inv = 1.0 / l; }
    let obase = (b * p.lq + qi) * p.d;
    for (var t: u32 = 0u; t < p.d; t = t + 1u) {
        o[obase + t] = acc[t] * inv;
    }
}
