// Micro-benchmark + correctness check for the Q8_0 mul_mm (prefill GEMM) paths:
//   - simdgroup kernel  (call_quantized_matmul_mm_t, Source::Quantized)
//   - matmul2d kernel   (call_quantized_matmul_mm_q8_0_mm2d, Source::QuantizedMM2d)
//
// Isolates the kernel from the engine (no model load, no decode, no sampler) so
// thermal noise and unrelated work don't pollute the comparison. Verifies the
// two kernels agree (max abs diff vs an f32 reference), then times each.
//
// Run on dbc:  cargo run --release --example q8_mm_bench
use half::f16;
use hanzo_metal_kernels::{
    metal::{create_command_buffer, CommandSemaphore, Device},
    GgmlDType, Kernels, RESOURCE_OPTIONS,
};
use std::sync::Arc;

const QK8_0: usize = 32;

// Quantize a row-major f32 matrix [rows x k] into Q8_0 blocks (d: f16, qs: i8[32]).
// Returns raw bytes: each block is 2 + 32 = 34 bytes.
fn quantize_q8_0(data: &[f32], rows: usize, k: usize) -> Vec<u8> {
    assert!(k % QK8_0 == 0);
    let blocks_per_row = k / QK8_0;
    let mut out = Vec::with_capacity(rows * blocks_per_row * (2 + QK8_0));
    for r in 0..rows {
        for b in 0..blocks_per_row {
            let base = r * k + b * QK8_0;
            let mut amax = 0f32;
            for i in 0..QK8_0 {
                amax = amax.max(data[base + i].abs());
            }
            let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            let dh = f16::from_f32(d);
            out.extend_from_slice(&dh.to_le_bytes());
            for i in 0..QK8_0 {
                let q = (data[base + i] * id).round().clamp(-127.0, 127.0) as i8;
                out.push(q as u8);
            }
        }
    }
    out
}

// CPU reference: C[n_act x n_w] in dst layout [act*n_w + w] = sum_k act[a,k]*w[w,k].
// Uses the *dequantized* weights (round-trip) so it matches what the GPU sees.
fn cpu_ref(weights_deq: &[f32], acts: &[f32], n_w: usize, n_act: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0f32; n_act * n_w];
    for a in 0..n_act {
        for w in 0..n_w {
            let mut acc = 0f32;
            for kk in 0..k {
                acc += acts[a * k + kk] * weights_deq[w * k + kk];
            }
            out[a * n_w + w] = acc;
        }
    }
    out
}

fn dequantize_q8_0(bytes: &[u8], rows: usize, k: usize) -> Vec<f32> {
    let blocks_per_row = k / QK8_0;
    let mut out = vec![0f32; rows * k];
    for r in 0..rows {
        for b in 0..blocks_per_row {
            let blk = (r * blocks_per_row + b) * (2 + QK8_0);
            let d = f16::from_le_bytes([bytes[blk], bytes[blk + 1]]).to_f32();
            for i in 0..QK8_0 {
                let q = bytes[blk + 2 + i] as i8;
                out[r * k + b * QK8_0 + i] = q as f32 * d;
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_case(device: &Device, kernels: &Kernels, n_w: usize, n_act: usize, k: usize, verify: bool) {
    let queue = device.new_command_queue().unwrap();
    let opts = RESOURCE_OPTIONS;

    // Deterministic pseudo-random inputs.
    let mut seed = 0x1234_5678u32;
    let mut rng = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((seed >> 8) as f32 / (1u32 << 24) as f32) - 0.5
    };
    let weights_f32: Vec<f32> = (0..n_w * k).map(|_| rng()).collect();
    let acts_f32: Vec<f32> = (0..n_act * k).map(|_| rng()).collect();

    let wq = quantize_q8_0(&weights_f32, n_w, k);
    let weights_deq = dequantize_q8_0(&wq, n_w, k);

    let w_buf = device
        .new_buffer_with_data(wq.as_ptr() as *const _, wq.len(), opts)
        .unwrap();
    let a_buf = device
        .new_buffer_with_data(
            acts_f32.as_ptr() as *const _,
            std::mem::size_of_val(&acts_f32[..]),
            opts,
        )
        .unwrap();
    let dst_simd = device.new_buffer(n_act * n_w * 4, opts).unwrap();
    let dst_mm2d = device.new_buffer(n_act * n_w * 4, opts).unwrap();

    // Argument shapes/strides matching call_quantized_matmul_mm_t's callers.
    let src0_shape = [1usize, 1, n_w, k];
    // src0_stride: hanzo_ml scales element strides by type_size/block_size (=34/32 for Q8_0).
    let ts_over_bs = 34f32 / 32f32;
    let src0_stride = [
        (n_w * k) as f32 * ts_over_bs,
        (n_w * k) as f32 * ts_over_bs,
        (k as f32 * ts_over_bs) as usize as f32,
        ts_over_bs,
    ]
    .map(|x| x as usize);
    let src1_shape = [1usize, 1, n_act, k];
    let src1_stride = [n_act * k * 4, n_act * k * 4, k * 4, 4]; // bytes
    let dst_shape = [1usize, n_act, n_w];

    // --- simdgroup kernel ---
    {
        let sem = Arc::new(CommandSemaphore::new());
        let cb = create_command_buffer(&queue, sem).unwrap();
        hanzo_metal_kernels::call_quantized_matmul_mm_t(
            device,
            &cb,
            kernels,
            GgmlDType::Q8_0,
            &src0_shape,
            &src0_stride,
            &w_buf,
            &src1_shape,
            &src1_stride,
            &a_buf,
            0,
            &dst_shape,
            0,
            &dst_simd,
        )
        .unwrap();
        cb.commit();
        cb.wait_until_completed();
    }
    // --- matmul2d kernel ---
    {
        let sem = Arc::new(CommandSemaphore::new());
        let cb = create_command_buffer(&queue, sem).unwrap();
        hanzo_metal_kernels::call_quantized_matmul_mm_q8_0_mm2d(
            device,
            &cb,
            kernels,
            &src0_shape,
            &src0_stride,
            &w_buf,
            &src1_shape,
            &src1_stride,
            &a_buf,
            0,
            &dst_shape,
            0,
            &dst_mm2d,
        )
        .unwrap();
        cb.commit();
        cb.wait_until_completed();
    }

    if verify {
        let reference = cpu_ref(&weights_deq, &acts_f32, n_w, n_act, k);
        let read = |buf: &hanzo_metal_kernels::metal::Buffer| -> Vec<f32> {
            let ptr = buf.contents() as *const f32;
            unsafe { std::slice::from_raw_parts(ptr, n_act * n_w).to_vec() }
        };
        let s = read(&dst_simd);
        let m = read(&dst_mm2d);
        let scale = reference.iter().fold(0f32, |a, &b| a.max(b.abs())).max(1e-6);
        let mut max_simd = 0f32;
        let mut max_mm2d = 0f32;
        for i in 0..n_act * n_w {
            max_simd = max_simd.max((s[i] - reference[i]).abs());
            max_mm2d = max_mm2d.max((m[i] - reference[i]).abs());
        }
        println!(
            "  verify n_w={n_w} n_act={n_act} k={k}: rel simd={:.2e} mm2d={:.2e}",
            max_simd / scale,
            max_mm2d / scale
        );
    }

    // --- timing ---
    let time_kernel = |which: u8| -> f64 {
        const WARMUP: usize = 3;
        const MIN_DUR: f64 = 2.0;
        let mut sum = 0f64;
        let mut iters = 0usize;
        for idx in 0.. {
            let sem = Arc::new(CommandSemaphore::new());
            let cb = create_command_buffer(&queue, sem).unwrap();
            let t0 = std::time::Instant::now();
            if which == 0 {
                hanzo_metal_kernels::call_quantized_matmul_mm_t(
                    device, &cb, kernels, GgmlDType::Q8_0, &src0_shape, &src0_stride, &w_buf,
                    &src1_shape, &src1_stride, &a_buf, 0, &dst_shape, 0, &dst_simd,
                )
                .unwrap();
            } else {
                hanzo_metal_kernels::call_quantized_matmul_mm_q8_0_mm2d(
                    device, &cb, kernels, &src0_shape, &src0_stride, &w_buf, &src1_shape,
                    &src1_stride, &a_buf, 0, &dst_shape, 0, &dst_mm2d,
                )
                .unwrap();
            }
            cb.commit();
            cb.wait_until_completed();
            let dt = t0.elapsed().as_secs_f64();
            if idx < WARMUP {
                continue;
            }
            sum += dt;
            iters += 1;
            if sum > MIN_DUR {
                break;
            }
        }
        let gflops = (2.0 * n_w as f64 * n_act as f64 * k as f64 * iters as f64) / (1e9 * sum);
        gflops
    };

    let g_simd = time_kernel(0);
    let g_mm2d = time_kernel(1);
    println!(
        "  n_w={n_w:6} n_act={n_act:5} k={k:5}  simd {g_simd:7.0} GF  mm2d {g_mm2d:7.0} GF ({:.2}x)",
        g_mm2d / g_simd
    );
}

// Small-batch (2..=8 activation cols) verify + timing for the mul_mv_ext Q8_0
// kernel vs the simdgroup GEMM. Compares both against the f32 reference and
// reports the per-call ms (this is the spec-decode / MTP verify step).
fn run_case_mv_ext(device: &Device, kernels: &Kernels, n_w: usize, n_act: usize, k: usize) {
    assert!(k % 128 == 0, "mv_ext requires k % 128 == 0");
    assert!((2..=8).contains(&n_act));
    let queue = device.new_command_queue().unwrap();
    let opts = RESOURCE_OPTIONS;

    let mut seed = 0x9E37_79B9u32;
    let mut rng = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((seed >> 8) as f32 / (1u32 << 24) as f32) - 0.5
    };
    let weights_f32: Vec<f32> = (0..n_w * k).map(|_| rng()).collect();
    let acts_f32: Vec<f32> = (0..n_act * k).map(|_| rng()).collect();

    let wq = quantize_q8_0(&weights_f32, n_w, k);
    let weights_deq = dequantize_q8_0(&wq, n_w, k);

    let w_buf = device
        .new_buffer_with_data(wq.as_ptr() as *const _, wq.len(), opts)
        .unwrap();
    let a_buf = device
        .new_buffer_with_data(
            acts_f32.as_ptr() as *const _,
            std::mem::size_of_val(&acts_f32[..]),
            opts,
        )
        .unwrap();
    let dst_ext = device.new_buffer(n_act * n_w * 4, opts).unwrap();
    let dst_simd = device.new_buffer(n_act * n_w * 4, opts).unwrap();

    let src0_shape = [1usize, 1, n_w, k];
    let ts_over_bs = 34f32 / 32f32;
    let src0_stride = [
        (n_w * k) as f32 * ts_over_bs,
        (n_w * k) as f32 * ts_over_bs,
        (k as f32 * ts_over_bs) as usize as f32,
        ts_over_bs,
    ]
    .map(|x| x as usize);
    let src1_shape = [1usize, 1, n_act, k];
    let src1_stride = [n_act * k * 4, n_act * k * 4, k * 4, 4]; // bytes
    let dst_shape = [1usize, n_act, n_w];

    let launch_ext = |cb: &hanzo_metal_kernels::metal::CommandBuffer| {
        hanzo_metal_kernels::call_quantized_matmul_mv_ext_q8_0(
            device, cb, kernels, &src0_shape, &src0_stride, &w_buf, &src1_shape, &src1_stride,
            &a_buf, 0, &dst_shape, 0, &dst_ext,
        )
        .unwrap();
    };
    let launch_simd = |cb: &hanzo_metal_kernels::metal::CommandBuffer| {
        hanzo_metal_kernels::call_quantized_matmul_mm_t(
            device, cb, kernels, GgmlDType::Q8_0, &src0_shape, &src0_stride, &w_buf, &src1_shape,
            &src1_stride, &a_buf, 0, &dst_shape, 0, &dst_simd,
        )
        .unwrap();
    };

    // correctness
    {
        let sem = Arc::new(CommandSemaphore::new());
        let cb = create_command_buffer(&queue, sem).unwrap();
        launch_ext(&cb);
        cb.commit();
        cb.wait_until_completed();
        let sem = Arc::new(CommandSemaphore::new());
        let cb = create_command_buffer(&queue, sem).unwrap();
        launch_simd(&cb);
        cb.commit();
        cb.wait_until_completed();
        let reference = cpu_ref(&weights_deq, &acts_f32, n_w, n_act, k);
        let read = |buf: &hanzo_metal_kernels::metal::Buffer| -> Vec<f32> {
            let ptr = buf.contents() as *const f32;
            unsafe { std::slice::from_raw_parts(ptr, n_act * n_w).to_vec() }
        };
        let e = read(&dst_ext);
        let s = read(&dst_simd);
        let scale = reference.iter().fold(0f32, |a, &b| a.max(b.abs())).max(1e-6);
        let mut max_ext = 0f32;
        let mut max_simd = 0f32;
        let mut max_es = 0f32;
        for i in 0..n_act * n_w {
            max_ext = max_ext.max((e[i] - reference[i]).abs());
            max_simd = max_simd.max((s[i] - reference[i]).abs());
            max_es = max_es.max((e[i] - s[i]).abs());
        }
        println!(
            "  verify n_w={n_w} n_act={n_act} k={k}: rel ext={:.2e} simd={:.2e} (ext-vs-simd {:.2e})",
            max_ext / scale,
            max_simd / scale,
            max_es / scale
        );
    }

    // timing (ms per call). Batch BATCH dispatches into one command buffer so the
    // per-submit overhead (~0.05-0.15ms) is amortized and we see kernel-execution
    // time, which is what compounds across the ~36-layer engine forward.
    let time = |which: u8| -> f64 {
        const WARMUP: usize = 3;
        const REPS: usize = 30;
        const BATCH: usize = 64;
        let mut sum = 0f64;
        for idx in 0..(WARMUP + REPS) {
            let sem = Arc::new(CommandSemaphore::new());
            let cb = create_command_buffer(&queue, sem).unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..BATCH {
                if which == 0 {
                    launch_ext(&cb);
                } else {
                    launch_simd(&cb);
                }
            }
            cb.commit();
            cb.wait_until_completed();
            let dt = t0.elapsed().as_secs_f64();
            if idx >= WARMUP {
                sum += dt;
            }
        }
        sum / (REPS * BATCH) as f64 * 1e3
    };
    let ms_ext = time(0);
    let ms_simd = time(1);
    println!(
        "  n_w={n_w:6} n_act={n_act:3} k={k:5}  ext {ms_ext:7.4} ms  simd {ms_simd:7.4} ms ({:.2}x faster)",
        ms_simd / ms_ext
    );
}

fn main() {
    let device = Device::system_default().unwrap();
    println!("device supports metal4: {}", device.supports_metal4());
    let kernels = Kernels::new();

    // mul_mv_ext (small-batch / spec-decode verify) correctness + timing.
    println!("== mul_mv_ext correctness + timing (small batch) ==");
    for &n_act in &[2usize, 3, 4, 5, 6, 8] {
        // attn proj shape; k=128 multiple required.
        run_case_mv_ext(&device, &kernels, 256, n_act, 128);
    }
    // Qwen3-8B-ish projection shapes at the verify batch (gamma+1 = 5).
    println!("== mul_mv_ext on Qwen3-8B shapes (n_act=5) ==");
    for &(n_w, k) in &[(4096usize, 4096usize), (12288, 4096), (4096, 12288), (1024, 4096)] {
        run_case_mv_ext(&device, &kernels, n_w, 5, k);
    }

    // Verify correctness on a few shapes first.
    println!("== correctness ==");
    for &(n_w, n_act, k) in &[(64usize, 32usize, 64usize), (256, 100, 128), (512, 575, 256)] {
        run_case(&device, &kernels, n_w, n_act, k, true);
    }

    // Throughput on Qwen3-8B-ish projection shapes at prefill batch sizes.
    // Qwen3-8B: hidden 4096, ffn ~12288, q/k/v/o around 4096/1024.
    println!("== throughput ==");
    let acts = [128usize, 575, 1024];
    let shapes = [
        (4096usize, 4096usize), // attn proj
        (12288, 4096),          // ffn up/gate (out=12288, k=4096)
        (4096, 12288),          // ffn down (out=4096, k=12288)
    ];
    for &n_act in &acts {
        for &(n_w, k) in &shapes {
            run_case(&device, &kernels, n_w, n_act, k, false);
        }
    }
}
