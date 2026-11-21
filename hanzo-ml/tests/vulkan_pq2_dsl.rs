//! End-to-end Vulkan qualification for production DSL `PQ2_0` and `PTQ1_0` dispatch in `hanzo-ml`.
//!
//! Covers:
//! 1. `PQ2_0` and `PTQ1_0` decode matvec (`m == 1`, `f32`) vs exact GGUF reference and legacy.
//! 2. `PQ2_0` and `PTQ1_0` prefill GEMM (`m > 1`, `bm = 32` and `bm = 64`, `f32`) vs exact GGUF
//!    reference, plus nonzero activation slice offset and 3D `[B, T, K]` batch shapes.
//! 3. Single-compile proof (`compiled 1x`) and dispatch counters (`quantized::dsl::stats`).
//! 4. Matched interleaved A/B benchmark (`#[ignore]`: decode + prefill, DSL vs legacy).
#![cfg(feature = "vulkan")]

use std::borrow::Cow;
use std::sync::Mutex;
use std::time::Instant;

use hanzo_ml::quantized::dsl::{self, QkImpl};
use hanzo_ml::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use hanzo_ml::{DType, Device, Module, Tensor};

static SERIAL: Mutex<()> = Mutex::new(());

fn act(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 7919) % 2003) as f32 / 1001.5) - 1.0)
        .collect()
}

fn raw_pq2(n: usize, k: usize, seed: u64) -> Vec<u8> {
    let (codes, d) = hanzo_kernel::ternary::pq2_0_synth(n, k, seed);
    hanzo_kernel::ternary::pq2_0_gguf_bytes(&codes, &d, n, k)
}

fn raw_ptq1(n: usize, k: usize, seed: u64) -> Vec<u8> {
    let (codes, d) = hanzo_kernel::ternary::ptq1_0_synth(n, k, seed);
    hanzo_kernel::ternary::ptq1_0_gguf_bytes(&codes, &d, n, k)
}

fn ref_matmul(dtype: GgmlDType, raw: &[u8], x: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(m * n);
    for row in 0..m {
        let xr = &x[row * k..(row + 1) * k];
        let yr = match dtype {
            GgmlDType::PQ2_0 => hanzo_kernel::ternary::pq2_0_gguf_ref(raw, xr, n, k),
            GgmlDType::PTQ1_0 => hanzo_kernel::ternary::ptq1_0_gguf_ref(raw, xr, n, k),
            _ => unreachable!(),
        };
        out.extend_from_slice(&yr);
    }
    out
}

fn matmul(dev: &Device, dtype: GgmlDType, raw: &[u8], n: usize, k: usize) -> QMatMul {
    let qt = QTensor::new(
        QStorage::from_data(Cow::Borrowed(raw), dev, dtype).unwrap(),
        (n, k),
    )
    .unwrap();
    QMatMul::from_qtensor(qt).unwrap()
}

fn scale_rel(want: &[f32], got: &[f32]) -> f32 {
    assert_eq!(want.len(), got.len());
    let maxref = want.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-30);
    want.iter()
        .zip(got)
        .map(|(a, b)| {
            assert!(b.is_finite(), "non-finite output {b}");
            (a - b).abs()
        })
        .fold(0f32, f32::max)
        / maxref
}

fn forward_as(mm: &QMatMul, x: &Tensor, dtype: GgmlDType, imp: QkImpl) -> Vec<f32> {
    match dtype {
        GgmlDType::PQ2_0 => dsl::set_pq2_impl(imp),
        GgmlDType::PTQ1_0 => dsl::set_ptq1_impl(imp),
        _ => unreachable!(),
    }
    mm.forward(x)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

fn dispatches_of(stats: &dsl::Stats, artifact: &str) -> u64 {
    stats
        .dispatches
        .iter()
        .filter(|(_, a, _, _, _)| *a == artifact)
        .map(|(_, _, _, _, c)| *c)
        .sum()
}

fn compiles_of(stats: &dsl::Stats, artifact: &str) -> u64 {
    stats
        .compiles
        .iter()
        .find(|(a, _)| a == artifact)
        .map(|(_, c)| *c)
        .unwrap_or(0)
}

#[test]
fn vulkan_pq2_and_ptq1_dsl_matvec_match_oracle_and_legacy() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dev = Device::new_vulkan(0).unwrap();
    dsl::reset_stats();
    let shapes = [(1024usize, 5120usize), (5120, 5120), (17408, 5120)];

    for &(n, k) in &shapes {
        let xv = act(k);
        let x32 = Tensor::from_vec(xv.clone(), (1, k), &dev).unwrap();

        for dtype in [GgmlDType::PQ2_0, GgmlDType::PTQ1_0] {
            let raw = match dtype {
                GgmlDType::PQ2_0 => raw_pq2(n, k, (n * 17 + k) as u64),
                GgmlDType::PTQ1_0 => raw_ptq1(n, k, (n * 31 + k) as u64),
                _ => unreachable!(),
            };
            let mm = matmul(&dev, dtype, &raw, n, k);
            let want = ref_matmul(dtype, &raw, &xv, 1, n, k);
            let got_dsl = forward_as(&mm, &x32, dtype, QkImpl::Dsl);
            let got_dsl2 = forward_as(&mm, &x32, dtype, QkImpl::Dsl);
            let got_leg = forward_as(&mm, &x32, dtype, QkImpl::Legacy);
            assert_eq!(got_dsl, got_dsl2, "Vulkan {dtype:?} non-deterministic");
            let rel_dsl = scale_rel(&want, &got_dsl);
            let rel_leg = scale_rel(&want, &got_leg);
            eprintln!(
                "Vulkan {dtype:?} matvec {n:>5}x{k:<5} dsl={rel_dsl:.2e} legacy={rel_leg:.2e}"
            );
            assert!(rel_dsl <= 5e-5, "Vulkan {dtype:?} matvec rel={rel_dsl:.3e}");
            assert!(rel_leg <= 5e-5, "Vulkan {dtype:?} legacy rel={rel_leg:.3e}");
        }
    }

    let st = dsl::stats();
    let pq2_art = dsl::pq2_matvec_artifact(5120, false).0;
    let ptq1_art = dsl::ptq1_matvec_artifact(5120, false).0;
    assert_eq!(dispatches_of(&st, pq2_art), 6);
    assert_eq!(dispatches_of(&st, ptq1_art), 6);
    assert_eq!(compiles_of(&st, pq2_art), 1);
    assert_eq!(compiles_of(&st, ptq1_art), 1);
    dsl::set_pq2_impl(QkImpl::Legacy);
    dsl::set_ptq1_impl(QkImpl::Legacy);
}

#[test]
fn vulkan_pq2_and_ptq1_dsl_gemm_bm32_bm64_offset_and_batch() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dev = Device::new_vulkan(0).unwrap();
    dsl::reset_stats();

    for dtype in [GgmlDType::PQ2_0, GgmlDType::PTQ1_0] {
        // bm32 + bm64 shapes
        for &(m, n, k, expect_bm) in &[
            (17usize, 73usize, 256usize, 32usize),
            (48, 256, 512, 32),
            (257, 2048, 256, 64),
        ] {
            let raw = match dtype {
                GgmlDType::PQ2_0 => raw_pq2(n, k, (m * 97 + n * 13 + k) as u64),
                GgmlDType::PTQ1_0 => raw_ptq1(n, k, (m * 101 + n * 19 + k) as u64),
                _ => unreachable!(),
            };
            let mm = matmul(&dev, dtype, &raw, n, k);
            let xv = act(m * k);
            let x = Tensor::from_vec(xv.clone(), (m, k), &dev).unwrap();
            let want = ref_matmul(dtype, &raw, &xv, m, n, k);

            let (art, bm) = match dtype {
                GgmlDType::PQ2_0 => dsl::pq2_matmul_artifact(m, n, false),
                GgmlDType::PTQ1_0 => dsl::ptq1_matmul_artifact(m, n, false),
                _ => unreachable!(),
            };
            assert_eq!(bm, expect_bm);

            let got_dsl = forward_as(&mm, &x, dtype, QkImpl::Dsl);
            let _ = forward_as(&mm, &x, dtype, QkImpl::Dsl);
            let rel_dsl = scale_rel(&want, &got_dsl);
            eprintln!(
                "Vulkan {dtype:?} gemm   m={m:>3} n={n:>4} k={k:<3} ({art}) rel={rel_dsl:.2e}"
            );
            assert!(
                rel_dsl <= 5e-3,
                "Vulkan {dtype:?} gemm ({art}) rel={rel_dsl:.3e}"
            );
        }

        // Nonzero start_offset (narrow) + 3D [B, T, K] batch
        let (n, k) = (96usize, 256usize);
        let raw = match dtype {
            GgmlDType::PQ2_0 => raw_pq2(n, k, 777),
            GgmlDType::PTQ1_0 => raw_ptq1(n, k, 888),
            _ => unreachable!(),
        };
        let mm = matmul(&dev, dtype, &raw, n, k);
        let x_big = act(6 * k);
        let x_full = Tensor::from_vec(x_big.clone(), (2, 3, k), &dev).unwrap();
        let x_slice = x_full.narrow(1, 2, 1).unwrap();
        let mut x_exp = Vec::with_capacity(2 * k);
        x_exp.extend_from_slice(&x_big[2 * k..3 * k]);
        x_exp.extend_from_slice(&x_big[5 * k..6 * k]);
        let want = ref_matmul(dtype, &raw, &x_exp, 2, n, k);

        match dtype {
            GgmlDType::PQ2_0 => dsl::set_pq2_impl(QkImpl::Dsl),
            GgmlDType::PTQ1_0 => dsl::set_ptq1_impl(QkImpl::Dsl),
            _ => unreachable!(),
        }
        let y = mm.forward(&x_slice).unwrap();
        assert_eq!(y.dims(), &[2, 1, n]);
        let got = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let rel = scale_rel(&want, &got);
        eprintln!("Vulkan {dtype:?} narrow+3D [2,1,{k}]->[2,1,{n}] rel={rel:.2e}");
        assert!(rel <= 5e-3, "Vulkan {dtype:?} narrow+3D rel={rel:.3e}");
    }

    let st = dsl::stats();
    for art in [
        "matmul_pq2_0_gguf_bm32_f32",
        "matmul_pq2_0_gguf_bm64_f32",
        "matmul_ptq1_0_gguf_bm32_f32",
        "matmul_ptq1_0_gguf_bm64_f32",
    ] {
        assert!(dispatches_of(&st, art) >= 2, "{art} did not dispatch");
        assert_eq!(compiles_of(&st, art), 1, "{art} compiled != 1x");
    }
    dsl::set_pq2_impl(QkImpl::Legacy);
    dsl::set_ptq1_impl(QkImpl::Legacy);
}

#[test]
#[ignore]
fn vulkan_pq2_and_ptq1_dsl_vs_legacy_bench() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dev = Device::new_vulkan(0).unwrap();
    let shapes = [
        (5120usize, 5120usize),
        (17408, 5120),
        (5120, 17408),
        (1024, 5120),
    ];
    let ms = [1usize, 32, 128, 512, 2048];

    for dtype in [GgmlDType::PQ2_0, GgmlDType::PTQ1_0] {
        eprintln!("\n=== Vulkan GB10 Matched A/B ({dtype:?}, f32) ===");
        eprintln!(
            "{:>12} {:>5} {:>10} {:>10} {:>9} {:>9}",
            "NxK", "M", "DSL(us)", "LEG(us)", "Speedup", "DSL(TF)"
        );
        for &(n, k) in &shapes {
            let raw = match dtype {
                GgmlDType::PQ2_0 => raw_pq2(n, k, (n * 7 + k) as u64),
                GgmlDType::PTQ1_0 => raw_ptq1(n, k, (n * 11 + k) as u64),
                _ => unreachable!(),
            };
            let mm = matmul(&dev, dtype, &raw, n, k);
            for &m in &ms {
                let x = Tensor::from_vec(act(m * k), (m, k), &dev).unwrap();
                let iters = if m >= 512 { 5 } else { 15 };

                // Warmup + measure DSL
                match dtype {
                    GgmlDType::PQ2_0 => dsl::set_pq2_impl(QkImpl::Dsl),
                    GgmlDType::PTQ1_0 => dsl::set_ptq1_impl(QkImpl::Dsl),
                    _ => unreachable!(),
                }
                let _ = mm.forward(&x).unwrap();
                dev.synchronize().unwrap();
                let mut dsl_times = Vec::with_capacity(iters);
                for _ in 0..iters {
                    let t0 = Instant::now();
                    let _ = mm.forward(&x).unwrap();
                    dev.synchronize().unwrap();
                    dsl_times.push(t0.elapsed().as_secs_f64() * 1e6);
                }
                dsl_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let dsl_us = dsl_times[iters / 2];

                // Warmup + measure Legacy (dequant + dense matmul)
                match dtype {
                    GgmlDType::PQ2_0 => dsl::set_pq2_impl(QkImpl::Legacy),
                    GgmlDType::PTQ1_0 => dsl::set_ptq1_impl(QkImpl::Legacy),
                    _ => unreachable!(),
                }
                let _ = mm.forward(&x).unwrap();
                dev.synchronize().unwrap();
                let mut leg_times = Vec::with_capacity(iters);
                for _ in 0..iters {
                    let t0 = Instant::now();
                    let _ = mm.forward(&x).unwrap();
                    dev.synchronize().unwrap();
                    leg_times.push(t0.elapsed().as_secs_f64() * 1e6);
                }
                leg_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let leg_us = leg_times[iters / 2];

                let speedup = leg_us / dsl_us;
                let tflops = (2.0 * m as f64 * n as f64 * k as f64) / (dsl_us * 1e6);
                eprintln!(
                    "{:>12} {:>5} {:>10.1} {:>10.1} {:>8.2}x {:>9.2}",
                    format!("{n}x{k}"),
                    m,
                    dsl_us,
                    leg_us,
                    speedup,
                    tflops
                );
            }
        }
    }
    dsl::set_pq2_impl(QkImpl::Legacy);
    dsl::set_ptq1_impl(QkImpl::Legacy);
}
