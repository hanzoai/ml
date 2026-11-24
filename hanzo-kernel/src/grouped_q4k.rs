//! Grouped Q4_K expert projection over the existing device-side route plan.

use crate::prelude::*;
use half::f16;

#[device]
fn signed_dot4(a: u32, b: u32) -> i32 {
    let a = Vector::<i32, Const<4>>::cast_from(Vector::<i8, Const<4>>::reinterpret::<u32>(a));
    let b = Vector::<i32, Const<4>>::cast_from(Vector::<i8, Const<4>>::reinterpret::<u32>(b));
    a.dot(b)
}

#[device]
#[allow(clippy::too_many_arguments)]
fn project_slots(
    wq: &Array<Vector<u32, Const<4>>>,
    xq: &Array<Vector<u32, Const<4>>>,
    xd: &Array<f16>,
    slot_map: &Array<i32>,
    out: &mut Array<f16>,
    first: usize,
    col: usize,
    n: usize,
    k: usize,
    weight_row: usize,
    lane: usize,
    #[comptime] count: usize,
) {
    let blocks = k / 256usize;
    let mut slots = Array::<usize>::new(count);
    let mut acc = Array::<f32>::new(count);
    #[unroll]
    for r in 0usize..count {
        slots[r] = slot_map[first + r] as usize;
        acc[r] = 0.0f32;
    }

    let mut unit = lane;
    while unit < blocks * 4usize {
        let block = unit / 4usize;
        let sub = unit % 4usize;
        let base = (weight_row * blocks + block) * 9usize;
        let header = wq[base];
        let ds = Vector::<f16, Const<2>>::reinterpret::<u32>(header[0]);
        let d = f32::cast_from(ds[0]);
        let dmin = f32::cast_from(ds[1]);

        // Two adjacent scale/min bytes, one for each nibble of the Q4_K chunk.
        let shift = ((sub % 2usize) * 16usize) as u32;
        let raw_scale = header[1] >> shift;
        let raw_min = header[2] >> shift;
        let mut scale = raw_scale & 0x3f3fu32;
        let mut min = raw_min & 0x3f3fu32;
        if sub >= 2usize {
            let low_bits = header[3] >> shift;
            scale = (low_bits & 0x0f0fu32) | ((raw_scale >> 2u32) & 0x3030u32);
            min = ((low_bits >> 4u32) & 0x0f0fu32) | ((raw_min >> 2u32) & 0x3030u32);
        }
        let sc_lo = (scale & 63u32) as i32;
        let sc_hi = ((scale >> 8u32) & 63u32) as i32;
        let m_lo = (min & 63u32) as i32;
        let m_hi = ((min >> 8u32) & 63u32) as i32;

        let mut dot_lo = Array::<i32>::new(count);
        let mut dot_hi = Array::<i32>::new(count);
        let mut sum_lo = Array::<i32>::new(count);
        let mut sum_hi = Array::<i32>::new(count);
        #[unroll]
        for r in 0usize..count {
            dot_lo[r] = 0i32;
            dot_hi[r] = 0i32;
            sum_lo[r] = 0i32;
            sum_hi[r] = 0i32;
        }
        #[unroll]
        for chunk in 0usize..2usize {
            // Reuse the packed weight load across all four slots.
            let weights = wq[base + 1usize + sub * 2usize + chunk];
            #[unroll]
            for r in 0usize..count {
                let xb = slots[r] * (k / 16usize) + block * 16usize + sub * 4usize + chunk;
                let lo = xq[xb];
                let hi = xq[xb + 2usize];
                #[unroll]
                for i in 0usize..4usize {
                    let w = weights[i];
                    let wl = w & 0x0f0f0f0fu32;
                    let wh = (w >> 4u32) & 0x0f0f0f0fu32;
                    dot_lo[r] += signed_dot4(wl, lo[i]);
                    dot_hi[r] += signed_dot4(wh, hi[i]);
                    sum_lo[r] += signed_dot4(0x01010101u32, lo[i]);
                    sum_hi[r] += signed_dot4(0x01010101u32, hi[i]);
                }
            }
        }
        #[unroll]
        for r in 0usize..count {
            let sb = slots[r] * (k / 32usize) + block * 8usize + sub * 2usize;
            let lo = f32::cast_from(xd[sb]);
            let hi = f32::cast_from(xd[sb + 1usize]);
            let scaled = fma(
                lo,
                f32::cast_from(dot_lo[r] * sc_lo),
                hi * f32::cast_from(dot_hi[r] * sc_hi),
            );
            let offset = fma(
                lo,
                f32::cast_from(sum_lo[r] * m_lo),
                hi * f32::cast_from(sum_hi[r] * m_hi),
            );
            acc[r] += fma(d, scaled, -(dmin * offset));
        }
        unit += 32usize;
    }

    // Keep the reference HIP wave32 reduction order, including for the one-slot tail.
    #[unroll]
    for stage in 0u32..5u32 {
        let offset = 16u32 >> stage;
        #[unroll]
        for r in 0usize..count {
            acc[r] += plane_shuffle_down(acc[r], offset);
        }
    }
    if lane == 0usize {
        #[unroll]
        for r in 0usize..count {
            out[slots[r] * n + col] = f16::cast_from(acc[r]);
        }
    }
}

/// One wave32 per output column, four columns per 128-thread block.
/// Metadata is `[n, k, ceil(n / 4)]`. Each Q4_K block is nine u32x4 vectors.
#[kernel(targets(rocm), unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn moe_grouped_q4k_gfx1201(
    wq: &Array<Vector<u32, Const<4>>>,
    xq: &Array<Vector<u32, Const<4>>>,
    xd: &Array<f16>,
    slot_map: &Array<i32>,
    tile_expert: &Array<i32>,
    tile_pos0: &Array<i32>,
    tile_nrows: &Array<i32>,
    tile_count: &Array<i32>,
    meta: &Array<u32>,
    out: &mut Array<f16>,
) {
    let n = meta[0] as usize;
    let k = meta[1] as usize;
    let cols = meta[2] as usize;
    let group = CUBE_POS_X as usize;
    let thread = UNIT_POS_X as usize;
    let tile = group / cols;
    let col = (group % cols) * 4usize + thread / 32usize;
    if tile < tile_count[0] as usize && col < n {
        let lane = thread % 32usize;
        let weight_row = tile_expert[tile] as usize * n + col;
        let first = tile_pos0[tile] as usize;
        let rows = tile_nrows[tile] as usize;
        let mut row = 0usize;
        while row + 4usize <= rows {
            project_slots(
                wq,
                xq,
                xd,
                slot_map,
                out,
                first + row,
                col,
                n,
                k,
                weight_row,
                lane,
                4usize,
            );
            row += 4usize;
        }
        while row < rows {
            project_slots(
                wq,
                xq,
                xd,
                slot_map,
                out,
                first + row,
                col,
                n,
                k,
                weight_row,
                lane,
                1usize,
            );
            row += 1usize;
        }
    }
}
