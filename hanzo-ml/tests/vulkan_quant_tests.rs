//! Numerical validation of the native-GGML Vulkan quantized matvec kernels (Q4_0 / Q8_0 / Q4_K)
//! against the CPU reference, on the real Vulkan GPU.
//!
//! Each test:
//!   1. builds an f32 weight `W[nout, k]` and quantizes it to a GGML dtype on the CPU,
//!   2. dequantizes that QTensor back to f32 — the EXACT quantized weights the GPU kernel reads,
//!   3. uploads the raw GGML block bytes to the GPU (`upload_qweight`) and runs the matvec kernel,
//!   4. compares the GPU output to a host f64 matvec over the dequantized weights.
//!
//! The kernel reads the same quantized bytes as the reference dequant, so the only difference is
//! float accumulation order: agreement must be ~1e-3 relative. We also report the magnitude of the
//! quantization error itself (GPU vs the ORIGINAL f32 weight) for context.
//!
//! Skips cleanly (prints + returns) when no Vulkan GPU is present, so it is safe in CI without a
//! device. Requires the `vulkan` feature.
#![cfg(feature = "vulkan")]

use hanzo_ml::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use hanzo_ml::{Device, Module, Tensor};
use std::sync::Arc;

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

// Run one (dtype, nout, k) case end to end on the GPU and return error stats vs the CPU reference.
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
    let vk = dev.as_vulkan_device()?;
    let wq = vk.upload_qweight(&raw)?;
    let y_gpu: Vec<f32> = match dtype {
        GgmlDType::Q4_0 => vk.matvec_q4_0(&wq, &x_host, nout, k)?,
        GgmlDType::Q8_0 => vk.matvec_q8_0(&wq, &x_host, nout, k)?,
        GgmlDType::Q4K => vk.matvec_q4k(&wq, &x_host, nout, k)?,
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
        let g = y_gpu[n] as f64;
        max_abs = max_abs.max((g - ref_deq).abs() as f32);
        sse += (g - ref_deq) * (g - ref_deq);
        ref_sq += ref_deq * ref_deq;
        quant_max_abs = quant_max_abs.max((g - ref_orig).abs() as f32);
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

// Open the first Vulkan GPU, or None (skip) if there is no device.
fn gpu() -> Option<Device> {
    match Device::new_vulkan(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("[vulkan_quant_tests] no Vulkan GPU ({e}); skipping");
            None
        }
    }
}

// Representative Qwen3 hidden sizes. k must be a multiple of the block size; these all are.
const SHAPES: &[(usize, usize)] = &[
    (2048, 2048), // Qwen3 hidden
    (4096, 2048), // up-proj-ish (nout != k)
    (2048, 4096), // down-proj-ish
    (512, 256),   // small, exercises k == one Q4_K super-block
];

#[test]
fn vulkan_matvec_q4_0_matches_cpu() -> hanzo_ml::Result<()> {
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
fn vulkan_matvec_q8_0_matches_cpu() -> hanzo_ml::Result<()> {
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

#[test]
fn vulkan_matvec_q4k_matches_cpu() -> hanzo_ml::Result<()> {
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

// ---------------------------------------------------------------------------------------------
// End-to-end: the actual model path. Build a quantized QTensor ON the Vulkan device (exactly like
// the GGUF loader: QStorage::from_data on Device::Vulkan -> QTensor::new), wrap it in a QMatMul
// (which now routes Q4_0/Q8_0/Q4_K to the native VulkanQuant kernels), and run QMatMul::forward on
// a single-row Vulkan activation -- the decode hot path. Compare to the CPU QMatMul forward over
// the same quantized weights. This validates the wiring, not just the raw kernel.
fn end_to_end_case(dev: &Device, dtype: GgmlDType, nout: usize, k: usize) -> hanzo_ml::Result<f32> {
    let cpu = Device::Cpu;
    let w_host: Vec<f32> = (0..nout * k).map(|i| pseudo(i) * 0.5).collect();
    let x_host: Vec<f32> = (0..k).map(|i| pseudo(i + 7)).collect();

    // CPU quantized weight (quantized once); reuse it for both the raw bytes and the reference.
    let w_t = Tensor::from_vec(w_host, (nout, k), &cpu)?;
    let q_cpu = Arc::new(QTensor::quantize(&w_t, dtype)?);
    let bytes = q_cpu.data()?.into_owned();

    // Reference: dequantize the SAME quantized weights to f32 and do an f64 matvec. This is the
    // ground truth the GPU kernel targets (dequantize-then-dot). NOTE we deliberately do NOT use the
    // CPU `QMatMul::forward` here: for k-quants that path runs `vec_dot`, which quantizes the
    // ACTIVATION to int8 per block first -- a different (lossier) algorithm than the GPU's
    // f32-activation dequant matvec, so the two legitimately differ by ~1e-1, not a kernel bug.
    let w_deq: Vec<f32> = q_cpu.dequantize(&cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let mut y_ref = vec![0f64; nout];
    for n in 0..nout {
        let mut acc = 0f64;
        for j in 0..k {
            acc += w_deq[n * k + j] as f64 * x_host[j] as f64;
        }
        y_ref[n] = acc;
    }

    // Vulkan quantized weight built the loader way, then QMatMul (native VulkanQuant path).
    let qs_vk = QStorage::from_data(std::borrow::Cow::Owned(bytes), dev, dtype)?;
    let q_vk = QTensor::new(qs_vk, (nout, k))?;
    let qm_vk = QMatMul::from_qtensor(q_vk)?;
    // The native decode kernel triggers on a single-row input on the Vulkan device.
    let x_vk = Tensor::from_vec(x_host, (1, k), dev)?;
    let y_vk = qm_vk.forward(&x_vk)?.flatten_all()?.to_vec1::<f32>()?;

    assert_eq!(y_vk.len(), nout);
    let mut max_abs = 0f32;
    for n in 0..nout {
        max_abs = max_abs.max((y_vk[n] as f64 - y_ref[n]).abs() as f32);
    }
    Ok(max_abs)
}

#[test]
fn vulkan_qmatmul_forward_matches_cpu() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else { return Ok(()) };
    // (nout, k); k divisible by 256 so all three dtypes are exercised on the same shapes.
    for &(nout, k) in &[(2048usize, 2048usize), (4096, 2048), (512, 256)] {
        for dt in [GgmlDType::Q4_0, GgmlDType::Q8_0, GgmlDType::Q4K] {
            let max_abs = end_to_end_case(&dev, dt, nout, k)?;
            println!("QMatMul::forward {dt:?}  nout={nout:5} k={k:5}  GPU-vs-(dequant ref) max_abs={max_abs:.3e}");
            // GPU dequant-matvec vs the f64 dequant reference over identical quantized weights:
            // pure fp32 accumulation noise.
            assert!(
                max_abs < 1e-3,
                "QMatMul::forward {dt:?} GPU/ref mismatch too large: {max_abs}"
            );
        }
    }
    Ok(())
}
