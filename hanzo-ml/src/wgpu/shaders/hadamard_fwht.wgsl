struct Params {
    x_off: u32,
    s_off: u32,
    block_size: u32,
    width: u32,
    has_signs: u32,
    is_inverse: u32,
    total_blocks: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> signs: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> smem: array<f32, 4096>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let tid = lid.x;
    let e_cnt = max(p.block_size >> 8u, 1u);
    let threads = p.block_size / e_cnt;
    let chunk_e = min(e_cnt, 16u);
    let inv_sqrt_b = inverseSqrt(f32(p.block_size));
    let blocks_per_row = p.width / p.block_size;
    var v: array<f32, 32>;

    for (var blk: u32 = wid.x; blk < p.total_blocks; blk = blk + nwg.x) {
        let blk_base = p.x_off + blk * p.block_size;
        let out_base = blk * p.block_size;
        let row_col_base = p.s_off + (blk % blocks_per_row) * p.block_size;

        if (tid < threads) {
            for (var r: u32 = 0u; r < e_cnt; r = r + 1u) {
                let pos = (r << 8u) + tid;
                var val = x[blk_base + pos];
                if (p.is_inverse == 0u && p.has_signs != 0u) {
                    val = val * signs[row_col_base + pos];
                }
                v[r] = val;
            }
        }

        for (var r0: u32 = 0u; r0 < e_cnt; r0 = r0 + chunk_e) {
            if (tid < threads) {
                for (var r: u32 = 0u; r < chunk_e; r = r + 1u) {
                    smem[(r << 8u) + tid] = v[r0 + r];
                }
            }
            workgroupBarrier();
            for (var stride: u32 = 1u; stride < threads; stride = stride << 1u) {
                var nv: array<f32, 16>;
                if (tid < threads) {
                    for (var r: u32 = 0u; r < chunk_e; r = r + 1u) {
                        let pos = (r << 8u) + tid;
                        let partner = smem[pos ^ stride];
                        if ((pos & stride) != 0u) {
                            nv[r] = partner - v[r0 + r];
                        } else {
                            nv[r] = v[r0 + r] + partner;
                        }
                    }
                }
                workgroupBarrier();
                if (tid < threads) {
                    for (var r: u32 = 0u; r < chunk_e; r = r + 1u) {
                        v[r0 + r] = nv[r];
                        smem[(r << 8u) + tid] = nv[r];
                    }
                }
                workgroupBarrier();
            }
        }

        if (tid < threads) {
            for (var sr: u32 = 1u; sr < e_cnt; sr = sr << 1u) {
                for (var base: u32 = 0u; base < e_cnt; base = base + 2u * sr) {
                    for (var off: u32 = 0u; off < sr; off = off + 1u) {
                        let a = v[base + off];
                        let b = v[base + off + sr];
                        v[base + off] = a + b;
                        v[base + off + sr] = a - b;
                    }
                }
            }
            for (var r: u32 = 0u; r < e_cnt; r = r + 1u) {
                let pos = (r << 8u) + tid;
                var val = v[r] * inv_sqrt_b;
                if (p.is_inverse != 0u && p.has_signs != 0u) {
                    val = val * signs[row_col_base + pos];
                }
                y[out_base + pos] = val;
            }
        }
        workgroupBarrier();
    }
}
