//! Numerical validation of the wgpu/WGSL compute backend (Milestone 1) against an f64 CPU
//! reference, on the real GB10 (picked via the Vulkan backend by wgpu's adapter enumeration).
//!
//! Milestone 1 covers three kernels:
//!   (a) f32 dense matmul  (Tensor::matmul on the wgpu device),
//!   (b) Q8_0 native-GGML quant matvec  (device.matvec_q8_0 + the end-to-end QMatMul path),
//!   (c) Q4_0 native-GGML quant matvec  (device.matvec_q4_0 + the end-to-end QMatMul path).
//!
//! Each quant case: builds an f32 weight W[nout,k], quantizes it on the CPU, dequantizes that
//! QTensor back to f32 (the EXACT quantized weights the GPU kernel reads), uploads the raw GGML
//! block bytes, runs the matvec, and compares to a host f64 matvec over the dequantized weights.
//! The kernel reads the same bytes as the reference dequant, so the only difference is float
//! accumulation order: agreement must be max_abs < 1e-3 AND max_rel(RMS) < 1e-3.
//!
//! Skips cleanly (prints + returns) when no wgpu GPU is present, so it is safe in CI without a
//! device. Requires the `wgpu` feature.
#![cfg(feature = "wgpu")]

use hanzo_ml::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use hanzo_ml::{Device, Module, Tensor};

// Deterministic pseudo-random f32 in [-1, 1) from a counter (no rng dep; reproducible).
fn pseudo(i: usize) -> f32 {
    // splitmix64-ish; take 24 mantissa bits.
    let mut z = (i as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
}

struct ErrStats {
    max_abs: f32,
    max_rel: f32,
    rms: f32,
    quant_max_abs: f32, // GPU vs original f32 weight (the quantization error itself)
}

// Run one (dtype, nout, k) matvec case end to end on the GPU and return error stats vs the CPU ref.
fn run_case(dev: &Device, dtype: GgmlDType, nout: usize, k: usize) -> hanzo_ml::Result<ErrStats> {
    // 1. f32 weight W[nout, k] and activation x[k].
    let w_host: Vec<f32> = (0..nout * k).map(|i| pseudo(i) * 0.5).collect();
    let x_host: Vec<f32> = (0..k).map(|i| pseudo(i + 1_000_003)).collect();

    // 2. Quantize on CPU, then dequantize to get the EXACT weights the kernel sees.
    let cpu = Device::Cpu;
    let w_t = Tensor::from_vec(w_host.clone(), (nout, k), &cpu)?;
    let q = QTensor::quantize(&w_t, dtype)?;
    let w_deq: Vec<f32> = q.dequantize(&cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let raw = q.data()?; // raw GGML block bytes

    // 3. Upload bytes + run the GPU kernel.
    let g = dev.as_wgpu_device()?;
    let wq = g.upload_qweight(&raw)?;
    let y_gpu: Vec<f32> = match dtype {
        GgmlDType::Q4_0 => g.matvec_q4_0(&wq, &x_host, nout, k)?,
        GgmlDType::Q8_0 => g.matvec_q8_0(&wq, &x_host, nout, k)?,
        GgmlDType::Q4K => g.matvec_q4k(&wq, &x_host, nout, k)?,
        _ => panic!("unsupported dtype in run_case: {dtype:?}"),
    };
    assert_eq!(y_gpu.len(), nout);

    // 4a. Reference: host f64 matvec over the DEQUANTIZED weights (what the kernel approximates).
    // 4b. Also matvec over the ORIGINAL weights to report the quantization error magnitude.
    let mut max_abs = 0f32;
    let mut sse = 0f64;
    let mut ref_sq = 0f64; // sum of ref^2, for an RMS-normalized relative error
    let mut quant_max_abs = 0f32;
    for n in 0..nout {
        let mut ref_deq = 0f64;
        let mut ref_orig = 0f64;
        for j in 0..k {
            ref_deq += w_deq[n * k + j] as f64 * x_host[j] as f64;
            ref_orig += w_host[n * k + j] as f64 * x_host[j] as f64;
        }
        let gout = y_gpu[n] as f64;
        max_abs = max_abs.max((gout - ref_deq).abs() as f32);
        sse += (gout - ref_deq) * (gout - ref_deq);
        ref_sq += ref_deq * ref_deq;
        quant_max_abs = quant_max_abs.max((gout - ref_orig).abs() as f32);
    }
    // Relative error normalized by the RMS of the reference outputs (the robust metric: a per-
    // element relative error explodes whenever a single dot-product lands near zero, which is a
    // property of the data, not the kernel). Both that and max_abs must be tiny.
    let ref_rms = (ref_sq / nout as f64).sqrt();
    let err_rms = (sse / nout as f64).sqrt();
    let max_rel = (err_rms / ref_rms.max(1e-9)) as f32;
    Ok(ErrStats {
        max_abs,
        max_rel,
        rms: err_rms as f32,
        quant_max_abs,
    })
}

// Open the first wgpu GPU, or None (skip) if there is no device.
fn gpu() -> Option<Device> {
    match Device::new_wgpu(0) {
        Ok(d) => {
            if let Ok(g) = d.as_wgpu_device() {
                eprintln!(
                    "[wgpu_quant_tests] selected adapter: {}",
                    g.adapter_description()
                );
            }
            Some(d)
        }
        Err(e) => {
            eprintln!("[wgpu_quant_tests] no wgpu GPU ({e}); skipping");
            None
        }
    }
}

// Representative Qwen3 hidden sizes. k must be a multiple of the block size (32 for Q4_0/Q8_0).
const SHAPES: &[(usize, usize)] = &[
    (2048, 2048), // Qwen3 hidden
    (4096, 2048), // up-proj-ish (nout != k)
    (2048, 4096), // down-proj-ish
    (512, 256),   // small
];

// (a) f32 dense matmul: C[m,n] = A[m,k] * B[k,n] on the wgpu device vs an f64 CPU reference.
fn matmul_case(dev: &Device, m: usize, k: usize, n: usize) -> hanzo_ml::Result<ErrStats> {
    let a_host: Vec<f32> = (0..m * k).map(|i| pseudo(i) * 0.5).collect();
    let b_host: Vec<f32> = (0..k * n).map(|i| pseudo(i + 777) * 0.5).collect();

    let a = Tensor::from_vec(a_host.clone(), (m, k), dev)?;
    let b = Tensor::from_vec(b_host.clone(), (k, n), dev)?;
    let c_gpu: Vec<f32> = a.matmul(&b)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(c_gpu.len(), m * n);

    let mut max_abs = 0f32;
    let mut sse = 0f64;
    let mut ref_sq = 0f64;
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0f64;
            for i in 0..k {
                acc += a_host[row * k + i] as f64 * b_host[i * n + col] as f64;
            }
            let gout = c_gpu[row * n + col] as f64;
            max_abs = max_abs.max((gout - acc).abs() as f32);
            sse += (gout - acc) * (gout - acc);
            ref_sq += acc * acc;
        }
    }
    let ref_rms = (ref_sq / (m * n) as f64).sqrt();
    let err_rms = (sse / (m * n) as f64).sqrt();
    let max_rel = (err_rms / ref_rms.max(1e-9)) as f32;
    Ok(ErrStats {
        max_abs,
        max_rel,
        rms: err_rms as f32,
        quant_max_abs: 0.0,
    })
}

#[test]
fn wgpu_matmul_f32_matches_cpu() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else { return Ok(()) };
    // (m, k, n). Includes non-square and a non-multiple-of-16 case to exercise the bounds checks.
    let cases = [
        (64usize, 64usize, 64usize),
        (128, 256, 64),
        (96, 128, 80),
        (17, 33, 19), // ragged: not a multiple of the 16x16 workgroup
    ];
    for &(m, k, n) in &cases {
        let s = matmul_case(&dev, m, k, n)?;
        println!(
            "f32 matmul m={m:4} k={k:4} n={n:4}  max_abs={:.3e} max_rel={:.3e} rms={:.3e}",
            s.max_abs, s.max_rel, s.rms
        );
        assert!(
            s.max_rel < 1e-3 && s.max_abs < 1e-3,
            "f32 matmul GPU/CPU mismatch too large: max_abs={} max_rel={}",
            s.max_abs,
            s.max_rel
        );
    }
    Ok(())
}

#[test]
fn wgpu_matvec_q4_0_matches_cpu() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else { return Ok(()) };
    for &(nout, k) in SHAPES {
        let s = run_case(&dev, GgmlDType::Q4_0, nout, k)?;
        println!(
            "Q4_0  nout={nout:5} k={k:5}  max_abs={:.3e} max_rel={:.3e} rms={:.3e}  (quant err vs f32: {:.3e})",
            s.max_abs, s.max_rel, s.rms, s.quant_max_abs
        );
        // GPU vs CPU-dequant reference: same bytes, only fp-order differences.
        assert!(
            s.max_rel < 1e-3 && s.max_abs < 1e-3,
            "Q4_0 GPU/CPU mismatch too large: max_abs={} max_rel={}",
            s.max_abs,
            s.max_rel
        );
    }
    Ok(())
}

#[test]
fn wgpu_matvec_q8_0_matches_cpu() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else { return Ok(()) };
    for &(nout, k) in SHAPES {
        let s = run_case(&dev, GgmlDType::Q8_0, nout, k)?;
        println!(
            "Q8_0  nout={nout:5} k={k:5}  max_abs={:.3e} max_rel={:.3e} rms={:.3e}  (quant err vs f32: {:.3e})",
            s.max_abs, s.max_rel, s.rms, s.quant_max_abs
        );
        assert!(
            s.max_rel < 1e-3 && s.max_abs < 1e-3,
            "Q8_0 GPU/CPU mismatch too large: max_abs={} max_rel={}",
            s.max_abs,
            s.max_rel
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// End-to-end: the actual model path. Build a quantized QTensor ON the wgpu device (exactly like the
// GGUF loader: QStorage::from_data on Device::Wgpu -> QTensor::new), wrap it in a QMatMul (which
// routes Q4_0/Q8_0 to the native WgpuQuant kernels), and run QMatMul::forward on a single-row wgpu
// activation -- the decode hot path. Compare to an f64 dequant reference. Validates the wiring.
fn end_to_end_case(dev: &Device, dtype: GgmlDType, nout: usize, k: usize) -> hanzo_ml::Result<f32> {
    let cpu = Device::Cpu;
    let w_host: Vec<f32> = (0..nout * k).map(|i| pseudo(i) * 0.5).collect();
    let x_host: Vec<f32> = (0..k).map(|i| pseudo(i + 7)).collect();

    // CPU quantized weight (quantized once); reuse it for both the raw bytes and the reference.
    let w_t = Tensor::from_vec(w_host, (nout, k), &cpu)?;
    let q_cpu = QTensor::quantize(&w_t, dtype)?;
    let bytes = q_cpu.data()?.into_owned();

    // Reference: dequantize the SAME quantized weights to f32 and do an f64 matvec (dequant-then-dot,
    // the ground truth the GPU kernel targets).
    let w_deq: Vec<f32> = q_cpu.dequantize(&cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let mut y_ref = vec![0f64; nout];
    for n in 0..nout {
        let mut acc = 0f64;
        for j in 0..k {
            acc += w_deq[n * k + j] as f64 * x_host[j] as f64;
        }
        y_ref[n] = acc;
    }

    // wgpu quantized weight built the loader way, then QMatMul (native WgpuQuant path).
    let qs = QStorage::from_data(std::borrow::Cow::Owned(bytes), dev, dtype)?;
    let q = QTensor::new(qs, (nout, k))?;
    let qm = QMatMul::from_qtensor(q)?;
    // The native decode kernel triggers on a single-row input on the wgpu device.
    let x = Tensor::from_vec(x_host, (1, k), dev)?;
    let y: Vec<f32> = qm.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;

    assert_eq!(y.len(), nout);
    let mut max_abs = 0f32;
    for n in 0..nout {
        max_abs = max_abs.max((y[n] as f64 - y_ref[n]).abs() as f32);
    }
    Ok(max_abs)
}

#[test]
fn wgpu_matvec_q4k_matches_cpu() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else { return Ok(()) };
    for &(nout, k) in SHAPES {
        let s = run_case(&dev, GgmlDType::Q4K, nout, k)?;
        println!(
            "Q4_K  nout={nout:5} k={k:5}  max_abs={:.3e} max_rel={:.3e} rms={:.3e}  (quant err vs f32: {:.3e})",
            s.max_abs, s.max_rel, s.rms, s.quant_max_abs
        );
        assert!(
            s.max_rel < 1e-3 && s.max_abs < 1e-3,
            "Q4_K GPU/CPU mismatch too large: max_abs={} max_rel={}",
            s.max_abs,
            s.max_rel
        );
    }
    Ok(())
}

#[test]
fn wgpu_qmatmul_forward_matches_cpu() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else { return Ok(()) };
    // (nout, k); k divisible by 256 so all three dtypes are exercised on the same shapes.
    for &(nout, k) in &[(2048usize, 2048usize), (4096, 2048), (512, 256)] {
        for dt in [GgmlDType::Q4_0, GgmlDType::Q8_0, GgmlDType::Q4K] {
            let max_abs = end_to_end_case(&dev, dt, nout, k)?;
            println!("QMatMul::forward {dt:?}  nout={nout:5} k={k:5}  GPU-vs-(dequant ref) max_abs={max_abs:.3e}");
            assert!(
                max_abs < 1e-3,
                "QMatMul::forward {dt:?} GPU/ref mismatch too large: {max_abs}"
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// MoE: the fused grouped quant matvec. Build a quantized expert bank [E, n, k] on the wgpu device
// and call QTensor::indexed_moe_forward (the Qwen3-MoE path) -- one on-GPU dispatch gathers each
// routed slot's expert and computes its matvec -- and compare to a host f64 reference over the
// dequantized per-expert weights.
fn moe_case(
    dev: &Device,
    dtype: GgmlDType,
    e_cnt: usize,
    n: usize,
    k: usize,
    t: usize,
    topk: usize,
) -> hanzo_ml::Result<f32> {
    let cpu = Device::Cpu;
    let bank_host: Vec<f32> = (0..e_cnt * n * k).map(|i| pseudo(i) * 0.5).collect();
    let bank_t = Tensor::from_vec(bank_host, (e_cnt, n, k), &cpu)?;
    let q_bank = QTensor::quantize(&bank_t.reshape((e_cnt * n, k))?, dtype)?; // quantize per row
    let bank_deq: Vec<f32> = q_bank.dequantize(&cpu)?.flatten_all()?.to_vec1::<f32>()?; // [E*n*k]
    let bytes = q_bank.data()?.into_owned();

    let x_host: Vec<f32> = (0..t * topk * k).map(|i| pseudo(i + 11) * 0.7).collect();
    let ids_host: Vec<u32> = (0..t * topk)
        .map(|i| ((i * 7 + 3) % e_cnt) as u32)
        .collect();

    let qs = QStorage::from_data(std::borrow::Cow::Owned(bytes), dev, dtype)?;
    let q = QTensor::new(qs, (e_cnt, n, k))?;
    let x = Tensor::from_vec(x_host.clone(), (t, topk, k), dev)?;
    let ids = Tensor::from_vec(ids_host.clone(), (t, topk), dev)?;
    let y = q
        .indexed_moe_forward(&x, &ids)?
        .reshape((t * topk, n))?
        .to_vec2::<f32>()?;

    let mut max_abs = 0f32;
    for slot in 0..t * topk {
        let e = ids_host[slot] as usize;
        for r in 0..n {
            let wbase = (e * n + r) * k;
            let xbase = slot * k;
            let mut acc = 0f64;
            for j in 0..k {
                acc += bank_deq[wbase + j] as f64 * x_host[xbase + j] as f64;
            }
            max_abs = max_abs.max((y[slot][r] as f64 - acc).abs() as f32);
        }
    }
    Ok(max_abs)
}

#[test]
fn wgpu_moe_forward_matches_cpu() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else { return Ok(()) };
    // (e_cnt, n, k, t, topk). k divisible by 256 so all three dtypes work on the same shapes.
    let cases = [
        (8usize, 256usize, 256usize, 2usize, 2usize),
        (16, 512, 256, 3, 4),
        (4, 768, 512, 1, 2), // decode-like: single token, top-2
    ];
    for &(e_cnt, n, k, t, topk) in &cases {
        for dt in [GgmlDType::Q4_0, GgmlDType::Q8_0, GgmlDType::Q4K] {
            let max_abs = moe_case(&dev, dt, e_cnt, n, k, t, topk)?;
            println!(
                "MoE {dt:?}  E={e_cnt:3} n={n:4} k={k:4} t={t} topk={topk}  GPU-vs-(dequant ref) max_abs={max_abs:.3e}"
            );
            assert!(
                max_abs < 1e-3,
                "MoE {dt:?} GPU/ref mismatch too large: {max_abs}"
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Flash-attention: fused QK^T -> softmax -> V (online softmax) vs an eager f64 reference. The GPU
// kernel never materializes the [Lq, Lk] score matrix; the reference computes it explicitly.
fn flash_case(
    dev: &Device,
    bh: usize,
    lq: usize,
    lk: usize,
    d: usize,
    causal: bool,
) -> hanzo_ml::Result<f32> {
    let scale = 1.0f32 / (d as f32).sqrt();
    let q: Vec<f32> = (0..bh * lq * d).map(|i| pseudo(i) * 0.5).collect();
    let k: Vec<f32> = (0..bh * lk * d).map(|i| pseudo(i + 5) * 0.5).collect();
    let v: Vec<f32> = (0..bh * lk * d).map(|i| pseudo(i + 9) * 0.5).collect();

    let g = dev.as_wgpu_device()?;
    let out = g.flash_attn(&q, &k, &v, bh, lq, lk, d, scale, causal)?;
    assert_eq!(out.len(), bh * lq * d);

    let mut max_abs = 0f32;
    for b in 0..bh {
        for qi in 0..lq {
            let last = if causal { qi + (lk - lq) + 1 } else { lk };
            let last = last.min(lk);
            let mut sc = vec![0f64; last];
            let mut mx = f64::NEG_INFINITY;
            for (j, scj) in sc.iter_mut().enumerate() {
                let mut s = 0f64;
                for t in 0..d {
                    s += q[(b * lq + qi) * d + t] as f64 * k[(b * lk + j) * d + t] as f64;
                }
                *scj = s * scale as f64;
                mx = mx.max(*scj);
            }
            let mut denom = 0f64;
            for scj in sc.iter_mut() {
                *scj = (*scj - mx).exp();
                denom += *scj;
            }
            for t in 0..d {
                let mut acc = 0f64;
                for (j, &pp) in sc.iter().enumerate() {
                    acc += pp * v[(b * lk + j) * d + t] as f64;
                }
                acc /= denom;
                let gg = out[(b * lq + qi) * d + t] as f64;
                max_abs = max_abs.max((gg - acc).abs() as f32);
            }
        }
    }
    Ok(max_abs)
}

#[test]
fn wgpu_flash_attn_matches_cpu() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else { return Ok(()) };
    // (bh, lq, lk, d, causal). d = 128 is Qwen3's head dim.
    let cases = [
        (8usize, 16usize, 16usize, 128usize, false),
        (8, 16, 16, 128, true),
        (8, 1, 64, 128, true), // decode: single query over a 64-key cache
        (4, 7, 13, 64, false), // ragged Lq != Lk, smaller d
    ];
    for &(bh, lq, lk, d, causal) in &cases {
        let max_abs = flash_case(&dev, bh, lq, lk, d, causal)?;
        println!(
            "FlashAttn bh={bh} lq={lq:3} lk={lk:3} d={d:3} causal={causal}  GPU-vs-(eager ref) max_abs={max_abs:.3e}"
        );
        assert!(
            max_abs < 1e-4,
            "FlashAttn GPU/ref mismatch too large: {max_abs}"
        );
    }
    Ok(())
}
