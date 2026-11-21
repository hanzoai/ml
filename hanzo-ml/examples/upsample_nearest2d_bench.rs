// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: BSD-3-Clause OR Apache-2.0

//! CPU benchmark and bit-exactness check for [`Tensor::upsample_nearest2d`].

use hanzo_ml::{Device, Result, Tensor};
use std::time::{Duration, Instant};

fn reference_upsample_nearest2d(
    src: &[f32],
    shape: (usize, usize, usize, usize),
    dst_h: usize,
    dst_w: usize,
) -> Vec<f32> {
    let (b_sz, c, src_h, src_w) = shape;
    let scale_h = src_h as f64 / dst_h as f64;
    let scale_w = src_w as f64 / dst_w as f64;
    let mut out = vec![0.0f32; b_sz * c * dst_h * dst_w];
    for b in 0..b_sz {
        for ch in 0..c {
            let src_plane = &src[(b * c + ch) * src_h * src_w..];
            let dst_plane = &mut out[(b * c + ch) * dst_h * dst_w..];
            for h in 0..dst_h {
                let sh = usize::min(src_h - 1, (h as f64 * scale_h) as usize);
                let src_row = &src_plane[sh * src_w..][..src_w];
                let dst_row = &mut dst_plane[h * dst_w..][..dst_w];
                for (w, dst_cell) in dst_row.iter_mut().enumerate() {
                    let sw = usize::min(src_w - 1, (w as f64 * scale_w) as usize);
                    *dst_cell = src_row[sw];
                }
            }
        }
    }
    out
}

fn percentile_ms(samples: &mut [Duration], pct: f64) -> f64 {
    samples.sort_unstable();
    let idx = (((samples.len() - 1) as f64) * pct).round() as usize;
    samples[idx].as_secs_f64() * 1_000.0
}

fn bench_shape(
    label: &str,
    shape: (usize, usize, usize, usize),
    dst_h: usize,
    dst_w: usize,
    iters: usize,
) -> Result<()> {
    let (b, c, h, w) = shape;
    let numel = b * c * h * w;
    let src_data: Vec<f32> = (0..numel).map(|i| ((i % 97) as f32) * 0.125).collect();
    let tensor = Tensor::from_slice(&src_data, shape, &Device::Cpu)?;

    let expected = reference_upsample_nearest2d(&src_data, shape, dst_h, dst_w);
    let actual = tensor.upsample_nearest2d(dst_h, dst_w)?;
    let actual_vec = actual.flatten_all()?.to_vec1::<f32>()?;
    let max_abs_diff = actual_vec
        .iter()
        .zip(expected.iter())
        .fold(0.0f32, |acc, (a, b)| acc.max((a - b).abs()));
    assert_eq!(
        max_abs_diff, 0.0,
        "{label}: expected bit-exact output, got max_abs_diff={max_abs_diff}"
    );

    // Warmup
    let _ = reference_upsample_nearest2d(&src_data, shape, dst_h, dst_w);
    let _ = tensor.upsample_nearest2d(dst_h, dst_w)?;

    let mut ref_times = Vec::with_capacity(iters);
    let mut hanzo_times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let ref_out =
            std::hint::black_box(reference_upsample_nearest2d(&src_data, shape, dst_h, dst_w));
        ref_times.push(t0.elapsed());
        std::hint::black_box(ref_out);

        let t1 = Instant::now();
        let out = std::hint::black_box(tensor.upsample_nearest2d(dst_h, dst_w)?);
        hanzo_times.push(t1.elapsed());
        std::hint::black_box(out);
    }

    let ref_p50 = percentile_ms(&mut ref_times, 0.50);
    let ref_p95 = percentile_ms(&mut ref_times, 0.95);
    let hz_p50 = percentile_ms(&mut hanzo_times, 0.50);
    let hz_p95 = percentile_ms(&mut hanzo_times, 0.95);
    let speedup = ref_p50 / hz_p50.max(1e-9);
    println!(
        "{label}: ref p50={ref_p50:.3}ms (p95={ref_p95:.3}ms) | hanzo p50={hz_p50:.3}ms (p95={hz_p95:.3}ms) | speedup={speedup:.2}x | max_diff={max_abs_diff:.1}"
    );
    Ok(())
}

fn main() -> Result<()> {
    bench_shape(
        "upsample_2x (1x256x64x64->128x128)",
        (1, 256, 64, 64),
        128,
        128,
        15,
    )?;
    bench_shape(
        "upsample_3x (1x128x48x48->144x144)",
        (1, 128, 48, 48),
        144,
        144,
        15,
    )?;
    Ok(())
}
