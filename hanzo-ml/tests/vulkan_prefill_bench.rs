//! Prefill GEMM microbenchmark (Vulkan): the native quantized GEMM vs the old dequantize->f32 path.
//!
//! PREFILL (M>1, prompt processing) used to dequantize the whole weight to an f32 tensor every
//! forward, then run a dense matmul. The native path keeps the weight quantized in VRAM and runs the
//! `mul_mat_q*` GEMM (each weight block decoded once per output column, reused across all M rows). We
//! time both for the SAME (dtype, M, N, K) and report us/call, GFLOP/s, and the speedup.
//!
//! * BEFORE = `qtensor.dequantize(dev)` + `x.matmul(w^T)` + readback   (the old prefill `else` arm)
//! * AFTER  = `QMatMul::forward(x)`                       + readback   (the SHIPPED VulkanQuant path:
//!   the native GEMM for M <= the per-dtype gate (Q5_K/Q6_K: 64, else 128), dequant beyond)
//!
//! Both produce [M, N] and read it back to host, so the readback cancels in the ratio. The crossover
//! grid sweeps M past the gate so you see BOTH the native-GEMM win (M <= 128: 1-27x) and the gate
//! holding the shipped path at >= 1.0x once the dequant fallback takes over (the per-tile weight
//! re-read of the column kernel is what the gate routes around at large M).
//!
//! Run:  cargo test -p hanzo-ml --features vulkan --release --test vulkan_prefill_bench -- --ignored --nocapture
//!
//! Skips cleanly when no Vulkan GPU is present. `#[ignore]` so it is opt-in (it allocates large f32
//! weights for the dequant path).
#![cfg(feature = "vulkan")]

use hanzo_ml::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use hanzo_ml::{Device, Module, Tensor};
use std::time::Instant;

const WARMUP: usize = 3;
const ITERS: usize = 10;

fn pseudo(i: usize) -> f32 {
    let mut z = (i as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
}

fn median_us(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

// Time a closure that returns the [M*N] output, forcing GPU completion via the host readback inside.
fn time_us(
    iters: usize,
    mut f: impl FnMut() -> hanzo_ml::Result<Vec<f32>>,
) -> hanzo_ml::Result<f64> {
    let mut s = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let _ = f()?;
        s.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    Ok(median_us(s))
}

struct Case {
    before_us: f64,
    after_us: f64,
}

fn bench_case(
    dev: &Device,
    dtype: GgmlDType,
    m: usize,
    nout: usize,
    k: usize,
) -> hanzo_ml::Result<Case> {
    let cpu = Device::Cpu;
    let w_host: Vec<f32> = (0..nout * k).map(|i| pseudo(i) * 0.5).collect();
    let x_host: Vec<f32> = (0..m * k).map(|i| pseudo(i + 7)).collect();
    let w_t = Tensor::from_vec(w_host, (nout, k), &cpu)?;
    let bytes = QTensor::quantize(&w_t, dtype)?.data()?.into_owned();

    let x_vk = Tensor::from_vec(x_host, (m, k), dev)?;

    // BEFORE: a standalone Vulkan QTensor we dequantize + dense-matmul every iter (the old prefill).
    let qt_before = QTensor::new(
        QStorage::from_data(std::borrow::Cow::Owned(bytes.clone()), dev, dtype)?,
        (nout, k),
    )?;
    let before = || -> hanzo_ml::Result<Vec<f32>> {
        let w = qt_before.dequantize(dev)?; // [nout, k] f32 materialized in VRAM
        let y = x_vk.matmul(&w.t()?)?; // [m, k] @ [k, nout] = [m, nout]
        y.flatten_all()?.to_vec1::<f32>()
    };

    // AFTER: the native VulkanQuant GEMM via the real model path (QMatMul::forward on [M, k]).
    let qm_after = QMatMul::from_qtensor(QTensor::new(
        QStorage::from_data(std::borrow::Cow::Owned(bytes), dev, dtype)?,
        (nout, k),
    )?)?;
    let after = || -> hanzo_ml::Result<Vec<f32>> {
        qm_after.forward(&x_vk)?.flatten_all()?.to_vec1::<f32>()
    };

    for _ in 0..WARMUP {
        let _ = before()?;
        let _ = after()?;
    }
    let before_us = time_us(ITERS, before)?;
    let after_us = time_us(ITERS, after)?;
    Ok(Case {
        before_us,
        after_us,
    })
}

fn gflops(m: usize, nout: usize, k: usize, us: f64) -> f64 {
    (2.0 * m as f64 * nout as f64 * k as f64) / (us * 1e-6) / 1e9
}

fn run(dev: &Device, dtype: GgmlDType, m: usize, nout: usize, k: usize) -> hanzo_ml::Result<()> {
    let c = bench_case(dev, dtype, m, nout, k)?;
    let bg = gflops(m, nout, k, c.before_us);
    let ag = gflops(m, nout, k, c.after_us);
    let speedup = c.before_us / c.after_us;
    println!(
        "{:>5?}  M={:4} N={:6} K={:5} | dequant->f32 {:9.1}us {:7.1} GFLOP/s | forward(gated) {:9.1}us {:7.1} GFLOP/s | speedup {:5.2}x",
        dtype, m, nout, k, c.before_us, bg, c.after_us, ag, speedup
    );
    Ok(())
}

#[test]
#[ignore = "perf bench; run with --ignored --nocapture"]
fn vulkan_prefill_gemm_vs_dequant() -> hanzo_ml::Result<()> {
    let Some(dev) = (match Device::new_vulkan(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("[vulkan_prefill_bench] no Vulkan GPU ({e}); skipping");
            None
        }
    }) else {
        return Ok(());
    };

    let types = [
        GgmlDType::Q4_0,
        GgmlDType::Q8_0,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
    ];

    println!("\n== Headline: M=512, K=4096, N in {{4096, 11008}} ==");
    for &nout in &[4096usize, 11008] {
        for dt in types {
            run(&dev, dt, 512, nout, 4096)?;
        }
    }

    // Crossover grid: the column-per-invocation kernel re-reads the weight ceil(M/8) times, so it
    // wins decisively at low M then goes weight-bandwidth-bound. Pin the crossover across M and N
    // (the larger-N FFN weights cross earlier) to set the gate threshold from data.
    println!("\n== Crossover grid: K=4096, M in {{16,64,128,256,512}}, N in {{4096,11008}} ==");
    for dt in [
        GgmlDType::Q4K,
        GgmlDType::Q6K,
        GgmlDType::Q8_0,
        GgmlDType::Q4_0,
    ] {
        for &nout in &[4096usize, 11008] {
            for &m in &[16usize, 64, 128, 256, 512] {
                run(&dev, dt, m, nout, 4096)?;
            }
        }
    }

    Ok(())
}
