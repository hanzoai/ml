//! Clean GPU kernel microbenchmark, backend-parametric (Vulkan / wgpu).
//!
//! Same kernel set + method as the Spark (GB10) run, so EVO (AMD Radeon 8060S) numbers line up
//! column-for-column:
//!   * f32 dense matmul                4096 x 4096 x 4096
//!   * Q8_0 / Q4_0 / Q4_K quant matvec nout = k = 4096  (decode hot path: single activation row)
//!   * MoE fused quant matvec (Q4_K)   E=8, n=512, k=512, topk=2, t=256
//!   * flash attention (fused)         d=128, seq=512, bh=8, causal
//!
//! Method per kernel: build inputs once, WARMUP runs, then ITERS timed runs, report the MEDIAN as
//! us/call and the achieved GFLOP/s (2*work_flops / median_seconds). The matvec / flash kernels take
//! host f32 slices and upload internally, so a "call" is upload+dispatch+readback -- identical on both
//! backends, the real decode-step cost. The MoE QTensor is built once on the device; only the fused
//! `indexed_moe_forward` dispatch is timed.
//!
//! Run (Vulkan):  cargo test -p hanzo-ml --features vulkan --release --test spark_bench -- --nocapture
//! Run (wgpu):    cargo test -p hanzo-ml --features wgpu    --release --test spark_bench -- --nocapture
//!
//! Skips cleanly (prints + returns) when no GPU of the selected backend is present.

#![cfg(any(feature = "vulkan", feature = "wgpu"))]

use hanzo_ml::quantized::{GgmlDType, QStorage, QTensor};
use hanzo_ml::{Device, Tensor};
use std::time::Instant;

const WARMUP: usize = 5;
const ITERS: usize = 30;

// Deterministic pseudo-random f32 in [-1, 1) from a counter (no rng dep; reproducible, and identical
// to the validation tests so the kernels see the same data distribution).
fn pseudo(i: usize) -> f32 {
    let mut z = (i as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
}

// Median of a set of durations, in microseconds.
fn median_us(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

struct Row {
    name: &'static str,
    us: f64,
    gflops: f64,
    note: String,
}

fn report(name: &'static str, samples: Vec<f64>, flops: f64) -> Row {
    let us = median_us(samples);
    let gflops = if us > 0.0 {
        flops / (us * 1e-6) / 1e9
    } else {
        f64::NAN
    };
    Row {
        name,
        us,
        gflops,
        note: String::new(),
    }
}

fn skip_row(name: &'static str, note: &str) -> Row {
    Row {
        name,
        us: f64::NAN,
        gflops: f64::NAN,
        note: note.to_string(),
    }
}

// ---- backend selection: pick the device + the kernel-call closures for the active feature -------
//
// Both backends expose the *same* method surface (upload_qweight, matvec_q4_0/q8_0/q4k, flash_attn,
// Tensor::matmul, QTensor::indexed_moe_forward); we only differ in how the device is opened and
// described. Exactly one of `vulkan` / `wgpu` should be enabled for a given run.

#[cfg(feature = "vulkan")]
fn open_device() -> Option<(Device, String)> {
    match Device::new_vulkan(0) {
        Ok(d) => {
            // VulkanDevice has no public name accessor; the backend selects the first non-CPU
            // (non-llvmpipe) physical device. Report coopmat support as the distinguishing detail.
            let coop = d
                .as_vulkan_device()
                .ok()
                .and_then(|v| v.coopmat_info())
                .map(|(m, n, k)| format!("coopmat {m}x{n}x{k}"))
                .unwrap_or_else(|| "no coopmat (fp32 path)".to_string());
            Some((d, format!("Vulkan (ash) first non-CPU device [{coop}]")))
        }
        Err(e) => {
            eprintln!("[spark_bench] no Vulkan GPU ({e}); skipping");
            None
        }
    }
}

#[cfg(feature = "wgpu")]
fn open_device() -> Option<(Device, String)> {
    match Device::new_wgpu(0) {
        Ok(d) => {
            let desc = d
                .as_wgpu_device()
                .map(|g| g.adapter_description().to_string())
                .unwrap_or_else(|_| "wgpu (unknown adapter)".to_string());
            Some((d, desc))
        }
        Err(e) => {
            eprintln!("[spark_bench] no wgpu GPU ({e}); skipping");
            None
        }
    }
}

// matvec_q* and flash_attn are inherent methods on the concrete device type; dispatch through the
// active-feature accessor so the same body serves both backends.
#[cfg(feature = "vulkan")]
macro_rules! dev_q {
    ($d:expr) => {
        $d.as_vulkan_device()?
    };
}
#[cfg(feature = "wgpu")]
macro_rules! dev_q {
    ($d:expr) => {
        $d.as_wgpu_device()?
    };
}

// ---- f32 dense matmul: C[m,n] = A[m,k] * B[k,n], 4096^3 -----------------------------------------
fn bench_matmul(dev: &Device) -> hanzo_ml::Result<Row> {
    let (m, k, n) = (4096usize, 4096usize, 4096usize);
    let a = Tensor::from_vec((0..m * k).map(|i| pseudo(i) * 0.5).collect(), (m, k), dev)?;
    let b = Tensor::from_vec(
        (0..k * n).map(|i| pseudo(i + 777) * 0.5).collect(),
        (k, n),
        dev,
    )?;

    for _ in 0..WARMUP {
        let c = a.matmul(&b)?;
        let _ = c.to_device(&Device::Cpu)?; // force completion
    }
    let mut s = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        let c = a.matmul(&b)?;
        let _ = c.to_device(&Device::Cpu)?; // include readback so the dispatch actually completes
        s.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    // matmul FLOPs = 2*m*n*k.
    Ok(report(
        "f32_matmul_4096",
        s,
        2.0 * m as f64 * n as f64 * k as f64,
    ))
}

// ---- quant matvec: y[nout] = Wq * x[k], nout = k = 4096 (single decode row) ---------------------
fn bench_matvec(dev: &Device, dtype: GgmlDType, name: &'static str) -> hanzo_ml::Result<Row> {
    let (nout, k) = (4096usize, 4096usize);
    let cpu = Device::Cpu;
    let w: Vec<f32> = (0..nout * k).map(|i| pseudo(i) * 0.5).collect();
    let x: Vec<f32> = (0..k).map(|i| pseudo(i + 1_000_003)).collect();
    let w_t = Tensor::from_vec(w, (nout, k), &cpu)?;
    let q = QTensor::quantize(&w_t, dtype)?;
    let raw = q.data()?;

    let g = dev_q!(dev);
    let wq = g.upload_qweight(&raw)?; // weights uploaded once; activation re-uploaded per call

    // g is the concrete device type (VulkanDevice or WgpuDevice); call the inherent matvec directly.
    macro_rules! run {
        () => {
            match dtype {
                GgmlDType::Q4_0 => g.matvec_q4_0(&wq, &x, nout, k)?,
                GgmlDType::Q8_0 => g.matvec_q8_0(&wq, &x, nout, k)?,
                GgmlDType::Q4K => g.matvec_q4k(&wq, &x, nout, k)?,
                _ => unreachable!(),
            }
        };
    }

    for _ in 0..WARMUP {
        let _ = run!();
    }
    let mut s = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        let _ = run!();
        s.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    // matvec FLOPs = 2*nout*k.
    Ok(report(name, s, 2.0 * nout as f64 * k as f64))
}

// ---- MoE fused quant matvec (Q4_K): expert bank [E,n,k], t tokens, top-k routed -----------------
fn bench_moe(dev: &Device) -> hanzo_ml::Result<Row> {
    let dtype = GgmlDType::Q4K;
    let (e_cnt, n, k, t, topk) = (8usize, 512usize, 512usize, 256usize, 2usize);
    let cpu = Device::Cpu;
    let bank: Vec<f32> = (0..e_cnt * n * k).map(|i| pseudo(i) * 0.5).collect();
    let bank_t = Tensor::from_vec(bank, (e_cnt, n, k), &cpu)?;
    let q_bank = QTensor::quantize(&bank_t.reshape((e_cnt * n, k))?, dtype)?;
    let bytes = q_bank.data()?.into_owned();

    let qs = QStorage::from_data(std::borrow::Cow::Owned(bytes), dev, dtype)?;
    let qv = QTensor::new(qs, (e_cnt, n, k))?;
    let x: Vec<f32> = (0..t * topk * k).map(|i| pseudo(i + 11) * 0.7).collect();
    let ids: Vec<u32> = (0..t * topk)
        .map(|i| ((i * 7 + 3) % e_cnt) as u32)
        .collect();
    let x_t = Tensor::from_vec(x, (t, topk, k), dev)?;
    let ids_t = Tensor::from_vec(ids, (t, topk), dev)?;

    for _ in 0..WARMUP {
        let y = qv.indexed_moe_forward(&x_t, &ids_t)?;
        let _ = y.to_device(&Device::Cpu)?;
    }
    let mut s = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        let y = qv.indexed_moe_forward(&x_t, &ids_t)?;
        let _ = y.to_device(&Device::Cpu)?;
        s.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    // Useful FLOPs = (t*topk) slots, each a [n,k] matvec = 2*n*k.
    let flops = 2.0 * (t * topk) as f64 * n as f64 * k as f64;
    Ok(report("moe_q4k_E8_topk2_n512", s, flops))
}

// ---- flash attention (fused QK^T -> softmax -> V): d=128, seq=512 -------------------------------
fn bench_flash(dev: &Device) -> hanzo_ml::Result<Row> {
    let (bh, lq, lk, d, causal) = (8usize, 512usize, 512usize, 128usize, true);
    let scale = 1.0f32 / (d as f32).sqrt();
    let q: Vec<f32> = (0..bh * lq * d).map(|i| pseudo(i) * 0.5).collect();
    let kk: Vec<f32> = (0..bh * lk * d).map(|i| pseudo(i + 5) * 0.5).collect();
    let v: Vec<f32> = (0..bh * lk * d).map(|i| pseudo(i + 9) * 0.5).collect();

    let g = dev_q!(dev);

    for _ in 0..WARMUP {
        let _ = g.flash_attn(&q, &kk, &v, bh, lq, lk, d, scale, causal)?;
    }
    let mut s = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        let _ = g.flash_attn(&q, &kk, &v, bh, lq, lk, d, scale, causal)?;
        s.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    // FLOPs ~ QK^T (2*bh*lq*lk*d) + softmax*V (2*bh*lq*lk*d) = 4*bh*lq*lk*d.
    let flops = 4.0 * bh as f64 * lq as f64 * lk as f64 * d as f64;
    Ok(report("flash_attn_d128_s512", s, flops))
}

#[test]
fn spark_bench() -> hanzo_ml::Result<()> {
    let Some((dev, adapter)) = open_device() else {
        return Ok(());
    };
    let backend = if cfg!(feature = "vulkan") {
        "vulkan"
    } else {
        "wgpu"
    };
    println!(
        "=== spark_bench backend={backend} adapter=\"{adapter}\" warmup={WARMUP} iters={ITERS} ==="
    );

    let mut rows = Vec::new();

    // Each kernel guarded so one failure (e.g. an unsupported dtype) still reports the rest.
    match bench_matmul(&dev) {
        Ok(r) => rows.push(r),
        Err(e) => rows.push(skip_row("f32_matmul_4096", &format!("ERR: {e}"))),
    }
    for (dt, nm) in [
        (GgmlDType::Q8_0, "q8_0_matvec_4096"),
        (GgmlDType::Q4_0, "q4_0_matvec_4096"),
        (GgmlDType::Q4K, "q4k_matvec_4096"),
    ] {
        match bench_matvec(&dev, dt, nm) {
            Ok(r) => rows.push(r),
            Err(e) => rows.push(skip_row(nm, &format!("ERR: {e}"))),
        }
    }
    match bench_moe(&dev) {
        Ok(r) => rows.push(r),
        Err(e) => rows.push(skip_row("moe_q4k_E8_topk2_n512", &format!("ERR: {e}"))),
    }
    match bench_flash(&dev) {
        Ok(r) => rows.push(r),
        Err(e) => rows.push(skip_row("flash_attn_d128_s512", &format!("ERR: {e}"))),
    }

    println!(
        "{:<26}  {:>12}  {:>12}  {}",
        "kernel", "us/call", "GFLOP/s", "note"
    );
    println!("{}", "-".repeat(72));
    for r in &rows {
        if r.us.is_nan() {
            println!("{:<26}  {:>12}  {:>12}  {}", r.name, "-", "-", r.note);
        } else {
            println!(
                "{:<26}  {:>12.3}  {:>12.1}  {}",
                r.name, r.us, r.gflops, r.note
            );
        }
    }
    println!("=== end spark_bench backend={backend} ===");
    Ok(())
}
