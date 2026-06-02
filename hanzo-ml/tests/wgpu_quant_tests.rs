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
                eprintln!("[wgpu_quant_tests] selected adapter: {}", g.adapter_description());
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
fn wgpu_qmatmul_forward_matches_cpu() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else { return Ok(()) };
    for &(nout, k) in &[(2048usize, 2048usize), (4096, 2048), (512, 256)] {
        for dt in [GgmlDType::Q4_0, GgmlDType::Q8_0] {
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
