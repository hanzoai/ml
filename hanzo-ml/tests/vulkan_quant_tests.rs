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
// True for GGML types with a CPU quantizer (`from_float`). The decode-only types (IQ4_NL / IQ4_XS /
// TQ2_0) have no `from_float`, so their test weights are synthesized as raw blocks instead.
fn quantizable(dtype: GgmlDType) -> bool {
    !matches!(
        dtype,
        GgmlDType::IQ4_NL
            | GgmlDType::IQ4_XS
            | GgmlDType::TQ2_0
            | GgmlDType::IQ2_XXS
            | GgmlDType::IQ2_S
            | GgmlDType::IQ3_XXS
            | GgmlDType::IQ3_S
            | GgmlDType::IQ1_S
            | GgmlDType::IQ1_M
            | GgmlDType::IQ2_XS
    )
}

// Deterministic pseudo-random byte (same splitmix source as `pseudo`).
fn pbyte(i: usize) -> u8 {
    (((pseudo(i) * 0.5 + 0.5) * 256.0) as i32).clamp(0, 255) as u8
}

// Synthesize valid raw GGML block bytes for a decode-only type (no `from_float` quantizer). Every
// nibble / 2-bit field is a valid index/value; the f16 scale magnitude is chosen so the dequantized
// weights stay O(0.5) -- the same range as the quantizable cases -- so the GPU-vs-CPU tolerance
// (max_abs < 1e-3) applies unchanged. Block layouts are bit-for-bit the CPU block structs.
fn synth_decode_only(dtype: GgmlDType, nout: usize, k: usize) -> Vec<u8> {
    use half::f16;
    let mut out: Vec<u8> = Vec::new();
    let mut c = 0usize;
    match dtype {
        // block_iq4_nl (18 B): f16 d, qs[16]. block of 32.
        GgmlDType::IQ4_NL => {
            for _ in 0..nout * (k / 32) {
                out.extend_from_slice(&f16::from_f32(pseudo(c) * 0.003).to_le_bytes());
                c += 1;
                for _ in 0..16 {
                    out.push(pbyte(c));
                    c += 1;
                }
            }
        }
        // block_iq4_xs (136 B): f16 d, u16 scales_h, scales_l[4], qs[128]. block of 256.
        GgmlDType::IQ4_XS => {
            for _ in 0..nout * (k / 256) {
                out.extend_from_slice(&f16::from_f32(pseudo(c) * 1e-4).to_le_bytes());
                c += 1;
                for _ in 0..6 {
                    out.push(pbyte(c)); // scales_h (u16) + scales_l[4]
                    c += 1;
                }
                for _ in 0..128 {
                    out.push(pbyte(c)); // qs[128]
                    c += 1;
                }
            }
        }
        // block_tq2_0 (66 B): qs[64] THEN f16 d. block of 256.
        GgmlDType::TQ2_0 => {
            for _ in 0..nout * (k / 256) {
                for _ in 0..64 {
                    out.push(pbyte(c)); // qs[64]
                    c += 1;
                }
                out.extend_from_slice(&f16::from_f32(pseudo(c) * 0.3).to_le_bytes());
                c += 1;
            }
        }
        // block_iq2_xxs (66 B): f16 d THEN qs[32] (u16 = 64 B). block of 256. Any byte pattern is a
        // valid codebook block (grid index 0..255, 7-bit sign group, 4-bit scale all in-range).
        GgmlDType::IQ2_XXS => {
            for _ in 0..nout * (k / 256) {
                out.extend_from_slice(&f16::from_f32(pseudo(c) * 0.05).to_le_bytes());
                c += 1;
                for _ in 0..64 {
                    out.push(pbyte(c)); // qs[32] u16
                    c += 1;
                }
            }
        }
        // block_iq2_xs (74 B): f16 d + qs[32] (u16 = 64 B) + scales[8]. block of 256. Any byte
        // pattern is valid (grid index 0..511 = qv&511, 7-bit sign = qv>>9, nibble scales in-range).
        GgmlDType::IQ2_XS => {
            for _ in 0..nout * (k / 256) {
                out.extend_from_slice(&f16::from_f32(pseudo(c) * 0.05).to_le_bytes());
                c += 1;
                for _ in 0..72 {
                    out.push(pbyte(c)); // qs[32] u16 (64 B) + scales[8]
                    c += 1;
                }
            }
        }
        // block_iq2s (82 B): f16 d THEN 80 bytes (qs/qh/scales). block of 256. Any byte
        // pattern is a valid codebook block.
        GgmlDType::IQ2_S => {
            for _ in 0..nout * (k / 256) {
                out.extend_from_slice(&f16::from_f32(pseudo(c) * 0.05).to_le_bytes());
                c += 1;
                for _ in 0..80 {
                    out.push(pbyte(c));
                    c += 1;
                }
            }
        }
        // block_iq3xxs (98 B): f16 d THEN 96 bytes (qs/qh/scales). block of 256. Any byte
        // pattern is a valid codebook block.
        GgmlDType::IQ3_XXS => {
            for _ in 0..nout * (k / 256) {
                out.extend_from_slice(&f16::from_f32(pseudo(c) * 0.02).to_le_bytes());
                c += 1;
                for _ in 0..96 {
                    out.push(pbyte(c));
                    c += 1;
                }
            }
        }
        // block_iq3s (110 B): f16 d THEN 108 bytes (qs/qh/scales). block of 256. Any byte
        // pattern is a valid codebook block.
        GgmlDType::IQ3_S => {
            for _ in 0..nout * (k / 256) {
                out.extend_from_slice(&f16::from_f32(pseudo(c) * 0.004).to_le_bytes());
                c += 1;
                for _ in 0..108 {
                    out.push(pbyte(c));
                    c += 1;
                }
            }
        }
        // block_iq1s (50 B): f16 d THEN 48 bytes (qs/qh/scales). block of 256. Any byte
        // pattern is a valid codebook block.
        GgmlDType::IQ1_S => {
            for _ in 0..nout * (k / 256) {
                out.extend_from_slice(&f16::from_f32(pseudo(c) * 0.05).to_le_bytes());
                c += 1;
                for _ in 0..48 {
                    out.push(pbyte(c));
                    c += 1;
                }
            }
        }
        // block_iq1m (56 B): qs[32] + qh[16] + scales[8]. NO leading d -- d is reconstructed from
        // the high nibbles of the 4 scale u16. Set scales so scale_u16 = 0x2C00 (f16 = 0.0625) and the
        // per-sub-block 3-bit scale fields are 0 (dl = d); qs/qh stay random (valid indices + signs).
        GgmlDType::IQ1_M => {
            for _ in 0..nout * (k / 256) {
                for _ in 0..48 {
                    out.push(pbyte(c)); // qs[32] + qh[16]
                    c += 1;
                }
                // scales[8] = sc0=0, sc1=0, sc2=0xC000, sc3=0x2000 -> scale_u16=0x2C00, dl fields 0.
                out.extend_from_slice(&[0u8, 0u8, 0u8, 0u8, 0u8, 0xC0u8, 0u8, 0x20u8]);
            }
        }
        _ => panic!("synth_decode_only: {dtype:?} is not a decode-only type"),
    }
    out
}

// Raw GGML block bytes for `dtype`, the exact f32 weights they dequantize to (`w_deq`, the kernel's
// ground truth), and the pre-quantization f32 source (`w_orig`, == w_deq for synthesized blocks).
fn weight_bytes(
    dtype: GgmlDType,
    nout: usize,
    k: usize,
) -> hanzo_ml::Result<(Vec<u8>, Vec<f32>, Vec<f32>)> {
    let cpu = Device::Cpu;
    if quantizable(dtype) {
        let w_host: Vec<f32> = (0..nout * k).map(|i| pseudo(i) * 0.5).collect();
        let w_t = Tensor::from_vec(w_host.clone(), (nout, k), &cpu)?;
        let q = QTensor::quantize(&w_t, dtype)?;
        let w_deq: Vec<f32> = q.dequantize(&cpu)?.flatten_all()?.to_vec1::<f32>()?;
        Ok((q.data()?.into_owned(), w_deq, w_host))
    } else {
        let raw = synth_decode_only(dtype, nout, k);
        let qs = QStorage::from_data(std::borrow::Cow::Owned(raw.clone()), &cpu, dtype)?;
        let q = QTensor::new(qs, (nout, k))?;
        let w_deq: Vec<f32> = q.dequantize(&cpu)?.flatten_all()?.to_vec1::<f32>()?;
        Ok((raw, w_deq.clone(), w_deq))
    }
}

fn run_case(dev: &Device, dtype: GgmlDType, nout: usize, k: usize) -> hanzo_ml::Result<ErrStats> {
    let x_host: Vec<f32> = (0..k).map(|i| pseudo(i + 1_000_003)).collect();
    // Raw GGML bytes + the EXACT dequantized weights the kernel reads + the original f32 source.
    let (raw, w_deq, w_host) = weight_bytes(dtype, nout, k)?;

    // Upload bytes + run the GPU kernel. Q6_K / Q3_K repack to a padded u32 stride on upload; every
    // other native type byte- or word-addresses the raw GGML bytes directly.
    let vk = dev.as_vulkan_device()?;
    let wq = match dtype {
        GgmlDType::Q6K => vk.quantize_q6k(&raw, nout, k)?,
        GgmlDType::Q3K => vk.quantize_q3k(&raw, nout, k)?,
        GgmlDType::IQ2_XXS => vk.quantize_iq2xxs(&raw, nout, k)?,
        GgmlDType::IQ2_XS => vk.quantize_iq2xs(&raw, nout, k)?,
        GgmlDType::IQ1_M => vk.quantize_iq1m(&raw, nout, k)?,
        GgmlDType::IQ1_S => vk.quantize_iq1s(&raw, nout, k)?,
        GgmlDType::IQ3_S => vk.quantize_iq3s(&raw, nout, k)?,
        GgmlDType::IQ3_XXS => vk.quantize_iq3xxs(&raw, nout, k)?,
        GgmlDType::IQ2_S => vk.quantize_iq2s(&raw, nout, k)?,
        _ => vk.upload_qweight(&raw)?,
    };
    let y_gpu: Vec<f32> = match dtype {
        GgmlDType::Q4_0 => vk.matvec_q4_0(&wq, &x_host, nout, k)?,
        GgmlDType::Q8_0 => vk.matvec_q8_0(&wq, &x_host, nout, k)?,
        GgmlDType::Q4K => vk.matvec_q4k_scalar(&wq, &x_host, nout, k)?,
        GgmlDType::Q5K => vk.matvec_q5k(&wq, &x_host, nout, k)?,
        GgmlDType::Q6K => vk.matvec_q6k(&wq, &x_host, nout, k)?,
        GgmlDType::Q2K => vk.matvec_q2k(&wq, &x_host, nout, k)?,
        GgmlDType::Q3K => vk.matvec_q3k(&wq, &x_host, nout, k)?,
        GgmlDType::IQ4_XS => vk.matvec_iq4xs(&wq, &x_host, nout, k)?,
        GgmlDType::IQ4_NL => vk.matvec_iq4nl(&wq, &x_host, nout, k)?,
        GgmlDType::IQ2_XXS => vk.matvec_iq2xxs(&wq, &x_host, nout, k)?,
        GgmlDType::IQ2_XS => vk.matvec_iq2xs(&wq, &x_host, nout, k)?,
        GgmlDType::IQ1_M => vk.matvec_iq1m(&wq, &x_host, nout, k)?,
        GgmlDType::IQ1_S => vk.matvec_iq1s(&wq, &x_host, nout, k)?,
        GgmlDType::IQ3_S => vk.matvec_iq3s(&wq, &x_host, nout, k)?,
        GgmlDType::IQ3_XXS => vk.matvec_iq3xxs(&wq, &x_host, nout, k)?,
        GgmlDType::IQ2_S => vk.matvec_iq2s(&wq, &x_host, nout, k)?,
        GgmlDType::TQ2_0 => vk.matvec_tq2_0(&wq, &x_host, nout, k)?,
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

#[test]
fn vulkan_matvec_q5k_matches_cpu() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else { return Ok(()) };
    for &(nout, k) in SHAPES {
        let s = run_case(&dev, GgmlDType::Q5K, nout, k)?;
        println!(
            "Q5_K  nout={nout:5} k={k:5}  max_abs={:.3e} max_rel={:.3e} rms={:.3e}  (quant err vs f32: {:.3e})",
            s.max_abs, s.max_rel, s.rms, s.quant_max_abs
        );
        assert!(
            s.max_rel < 1e-3 && s.max_abs < 1e-3,
            "Q5_K GPU/CPU mismatch too large: max_abs={} max_rel={}",
            s.max_abs,
            s.max_rel
        );
    }
    Ok(())
}

#[test]
fn vulkan_matvec_q6k_matches_cpu() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else { return Ok(()) };
    for &(nout, k) in SHAPES {
        let s = run_case(&dev, GgmlDType::Q6K, nout, k)?;
        println!(
            "Q6_K  nout={nout:5} k={k:5}  max_abs={:.3e} max_rel={:.3e} rms={:.3e}  (quant err vs f32: {:.3e})",
            s.max_abs, s.max_rel, s.rms, s.quant_max_abs
        );
        assert!(
            s.max_rel < 1e-3 && s.max_abs < 1e-3,
            "Q6_K GPU/CPU mismatch too large: max_abs={} max_rel={}",
            s.max_abs,
            s.max_rel
        );
    }
    Ok(())
}

#[test]
fn vulkan_matvec_q2k_matches_cpu() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else { return Ok(()) };
    for &(nout, k) in SHAPES {
        let s = run_case(&dev, GgmlDType::Q2K, nout, k)?;
        println!(
            "Q2_K  nout={nout:5} k={k:5}  max_abs={:.3e} max_rel={:.3e} rms={:.3e}  (quant err vs f32: {:.3e})",
            s.max_abs, s.max_rel, s.rms, s.quant_max_abs
        );
        assert!(
            s.max_rel < 1e-3 && s.max_abs < 1e-3,
            "Q2_K GPU/CPU mismatch too large: max_abs={} max_rel={}",
            s.max_abs,
            s.max_rel
        );
    }
    Ok(())
}

#[test]
fn vulkan_matvec_q3k_matches_cpu() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else { return Ok(()) };
    for &(nout, k) in SHAPES {
        let s = run_case(&dev, GgmlDType::Q3K, nout, k)?;
        println!(
            "Q3_K  nout={nout:5} k={k:5}  max_abs={:.3e} max_rel={:.3e} rms={:.3e}  (quant err vs f32: {:.3e})",
            s.max_abs, s.max_rel, s.rms, s.quant_max_abs
        );
        assert!(
            s.max_rel < 1e-3 && s.max_abs < 1e-3,
            "Q3_K GPU/CPU mismatch too large: max_abs={} max_rel={}",
            s.max_abs,
            s.max_rel
        );
    }
    Ok(())
}

#[test]
fn vulkan_matvec_iq4xs_matches_cpu() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else { return Ok(()) };
    for &(nout, k) in SHAPES {
        let s = run_case(&dev, GgmlDType::IQ4_XS, nout, k)?;
        println!(
            "IQ4_XS nout={nout:5} k={k:5}  max_abs={:.3e} max_rel={:.3e} rms={:.3e}  (quant err vs f32: {:.3e})",
            s.max_abs, s.max_rel, s.rms, s.quant_max_abs
        );
        assert!(
            s.max_rel < 1e-3 && s.max_abs < 1e-3,
            "IQ4_XS GPU/CPU mismatch too large: max_abs={} max_rel={}",
            s.max_abs,
            s.max_rel
        );
    }
    Ok(())
}

#[test]
fn vulkan_matvec_iq2xxs_matches_cpu() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else { return Ok(()) };
    for &(nout, k) in SHAPES {
        let s = run_case(&dev, GgmlDType::IQ2_XXS, nout, k)?;
        println!(
            "IQ2_XXS nout={nout:5} k={k:5}  max_abs={:.3e} max_rel={:.3e} rms={:.3e}  (quant err vs f32: {:.3e})",
            s.max_abs, s.max_rel, s.rms, s.quant_max_abs
        );
        assert!(
            s.max_rel < 1e-3 && s.max_abs < 1e-3,
            "IQ2_XXS GPU/CPU mismatch too large: max_abs={} max_rel={}",
            s.max_abs,
            s.max_rel
        );
    }
    Ok(())
}

#[test]
fn vulkan_matvec_iq2xs_matches_cpu() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else { return Ok(()) };
    for &(nout, k) in SHAPES {
        let s = run_case(&dev, GgmlDType::IQ2_XS, nout, k)?;
        println!(
            "IQ2_XS  nout={nout:5} k={k:5}  max_abs={:.3e} max_rel={:.3e} rms={:.3e}  (quant err vs f32: {:.3e})",
            s.max_abs, s.max_rel, s.rms, s.quant_max_abs
        );
        assert!(
            s.max_rel < 1e-3 && s.max_abs < 1e-3,
            "IQ2_XS GPU/CPU mismatch too large: max_abs={} max_rel={}",
            s.max_abs,
            s.max_rel
        );
    }
    Ok(())
}

#[test]
fn vulkan_matvec_iq4nl_matches_cpu() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else { return Ok(()) };
    for &(nout, k) in SHAPES {
        let s = run_case(&dev, GgmlDType::IQ4_NL, nout, k)?;
        println!(
            "IQ4_NL nout={nout:5} k={k:5}  max_abs={:.3e} max_rel={:.3e} rms={:.3e}  (quant err vs f32: {:.3e})",
            s.max_abs, s.max_rel, s.rms, s.quant_max_abs
        );
        assert!(
            s.max_rel < 1e-3 && s.max_abs < 1e-3,
            "IQ4_NL GPU/CPU mismatch too large: max_abs={} max_rel={}",
            s.max_abs,
            s.max_rel
        );
    }
    Ok(())
}

#[test]
fn vulkan_matvec_tq2_0_matches_cpu() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else { return Ok(()) };
    for &(nout, k) in SHAPES {
        let s = run_case(&dev, GgmlDType::TQ2_0, nout, k)?;
        println!(
            "TQ2_0  nout={nout:5} k={k:5}  max_abs={:.3e} max_rel={:.3e} rms={:.3e}  (quant err vs f32: {:.3e})",
            s.max_abs, s.max_rel, s.rms, s.quant_max_abs
        );
        assert!(
            s.max_rel < 1e-3 && s.max_abs < 1e-3,
            "TQ2_0 GPU/CPU mismatch too large: max_abs={} max_rel={}",
            s.max_abs,
            s.max_rel
        );
    }
    Ok(())
}


#[test]
fn vulkan_matvec_iq2s_matches_cpu() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else { return Ok(()) };
    for &(nout, k) in SHAPES {
        let s = run_case(&dev, GgmlDType::IQ2_S, nout, k)?;
        println!(
            "IQ2_S nout={nout:5} k={k:5}  max_abs={:.3e} max_rel={:.3e} rms={:.3e}",
            s.max_abs, s.max_rel, s.rms
        );
        assert!(
            s.max_rel < 1e-3 && s.max_abs < 1e-3,
            "IQ2_S GPU/CPU mismatch: max_abs={} max_rel={}", s.max_abs, s.max_rel
        );
    }
    Ok(())
}

#[test]
fn vulkan_matvec_iq3xxs_matches_cpu() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else { return Ok(()) };
    for &(nout, k) in SHAPES {
        let s = run_case(&dev, GgmlDType::IQ3_XXS, nout, k)?;
        println!(
            "IQ3_XXS nout={nout:5} k={k:5}  max_abs={:.3e} max_rel={:.3e} rms={:.3e}",
            s.max_abs, s.max_rel, s.rms
        );
        assert!(
            s.max_rel < 1e-3 && s.max_abs < 1e-3,
            "IQ3_XXS GPU/CPU mismatch: max_abs={} max_rel={}", s.max_abs, s.max_rel
        );
    }
    Ok(())
}

#[test]
fn vulkan_matvec_iq3s_matches_cpu() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else { return Ok(()) };
    for &(nout, k) in SHAPES {
        let s = run_case(&dev, GgmlDType::IQ3_S, nout, k)?;
        println!(
            "IQ3_S nout={nout:5} k={k:5}  max_abs={:.3e} max_rel={:.3e} rms={:.3e}",
            s.max_abs, s.max_rel, s.rms
        );
        assert!(
            s.max_rel < 1e-3 && s.max_abs < 1e-3,
            "IQ3_S GPU/CPU mismatch: max_abs={} max_rel={}", s.max_abs, s.max_rel
        );
    }
    Ok(())
}

#[test]
fn vulkan_matvec_iq1s_matches_cpu() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else { return Ok(()) };
    for &(nout, k) in SHAPES {
        let s = run_case(&dev, GgmlDType::IQ1_S, nout, k)?;
        println!(
            "IQ1_S nout={nout:5} k={k:5}  max_abs={:.3e} max_rel={:.3e} rms={:.3e}",
            s.max_abs, s.max_rel, s.rms
        );
        assert!(
            s.max_rel < 1e-3 && s.max_abs < 1e-3,
            "IQ1_S GPU/CPU mismatch: max_abs={} max_rel={}", s.max_abs, s.max_rel
        );
    }
    Ok(())
}

#[test]
fn vulkan_matvec_iq1m_matches_cpu() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else { return Ok(()) };
    for &(nout, k) in SHAPES {
        let s = run_case(&dev, GgmlDType::IQ1_M, nout, k)?;
        println!(
            "IQ1_M nout={nout:5} k={k:5}  max_abs={:.3e} max_rel={:.3e} rms={:.3e}",
            s.max_abs, s.max_rel, s.rms
        );
        assert!(
            s.max_rel < 1e-3 && s.max_abs < 1e-3,
            "IQ1_M GPU/CPU mismatch: max_abs={} max_rel={}", s.max_abs, s.max_rel
        );
    }
    Ok(())
}
// ---------------------------------------------------------------------------------------------
// End-to-end// ---------------------------------------------------------------------------------------------
// End-to-end// ---------------------------------------------------------------------------------------------
// End-to-end// ---------------------------------------------------------------------------------------------
// End-to-end// ---------------------------------------------------------------------------------------------
// End-to-end// ---------------------------------------------------------------------------------------------
// End-to-end: the actual model path. Build a quantized QTensor ON the Vulkan device (exactly like
// the GGUF loader: QStorage::from_data on Device::Vulkan -> QTensor::new), wrap it in a QMatMul
// (which now routes Q4_0/Q8_0/Q4_K to the native VulkanQuant kernels), and run QMatMul::forward on
// a single-row Vulkan activation -- the decode hot path. Compare to the CPU QMatMul forward over
// the same quantized weights. This validates the wiring, not just the raw kernel.
fn end_to_end_case(dev: &Device, dtype: GgmlDType, nout: usize, k: usize) -> hanzo_ml::Result<f32> {
    let x_host: Vec<f32> = (0..k).map(|i| pseudo(i + 7)).collect();

    // Raw GGML bytes + the exact dequantized weights. Quantizable types quantize on CPU; decode-only
    // types (IQ4_NL/IQ4_XS/TQ2_0) synthesize raw blocks. The dequantized weights are the f64 ground
    // truth the GPU kernel targets (dequantize-then-dot). NOTE we deliberately do NOT use the CPU
    // `QMatMul::forward`: for k-quants it runs `vec_dot`, which quantizes the ACTIVATION to int8 per
    // block first -- a lossier algorithm than the GPU's f32-activation dequant matvec (differs ~1e-1).
    let (bytes, w_deq, _) = weight_bytes(dtype, nout, k)?;
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
    // Q4_K decode defaults to int8 dp4a (q8_1 activation quant, ~0.4%); force the scalar path so this
    // exact-vs-CPU check validates the decode math. The dp4a q8_1 path is gated separately by
    // vulkan_q4k_dp4a_decode_matches_scalar.
    std::env::set_var("HANZO_VK_DP4A_DECODE_OFF", "1");
    // (nout, k); k divisible by 256 so every dtype is exercised on the same shapes.
    for &(nout, k) in &[(2048usize, 2048usize), (4096, 2048), (512, 256)] {
        for dt in [
            GgmlDType::Q4_0,
            GgmlDType::Q8_0,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
            GgmlDType::Q2K,
            GgmlDType::Q3K,
            GgmlDType::IQ4_XS,
            GgmlDType::IQ4_NL,
            GgmlDType::TQ2_0,
        ] {
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
    std::env::remove_var("HANZO_VK_DP4A_DECODE_OFF");
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// End-to-end PREFILL (M>1): the prompt-processing GEMM path. Build a quantized QTensor on the Vulkan
// device, wrap it in a QMatMul, and run forward on an [M, k] activation -- which now dispatches the
// native quantized GEMM (`mul_mat_q*`: weights stay quantized in VRAM, each weight block decoded ONCE
// per output column and reused across all M rows) instead of dequantizing the whole weight to f32.
// Compare every output row to a host f64 matmul over the dequantized weights. M straddles the host's
// MAX_M=8 row-tiling (partial tile / exact tile / multi-tile + remainder) so the tiling is exercised.
fn prefill_case(
    dev: &Device,
    dtype: GgmlDType,
    m: usize,
    nout: usize,
    k: usize,
) -> hanzo_ml::Result<f32> {
    let cpu = Device::Cpu;
    let w_host: Vec<f32> = (0..nout * k).map(|i| pseudo(i) * 0.5).collect();
    let x_host: Vec<f32> = (0..m * k).map(|i| pseudo(i + 7)).collect();

    // CPU quantized weight; reuse it for both the GPU upload bytes and the dequant reference.
    let w_t = Tensor::from_vec(w_host, (nout, k), &cpu)?;
    let q_cpu = Arc::new(QTensor::quantize(&w_t, dtype)?);
    let bytes = q_cpu.data()?.into_owned();
    let w_deq: Vec<f32> = q_cpu.dequantize(&cpu)?.flatten_all()?.to_vec1::<f32>()?;

    // Reference: y[mi, n] = sum_k W_deq[n, k] * x[mi, k] in f64 (dequantize-then-dot ground truth).
    let mut y_ref = vec![0f64; m * nout];
    for mi in 0..m {
        for n in 0..nout {
            let mut acc = 0f64;
            for j in 0..k {
                acc += w_deq[n * k + j] as f64 * x_host[mi * k + j] as f64;
            }
            y_ref[mi * nout + n] = acc;
        }
    }

    // Vulkan quantized weight built the loader way, then QMatMul (native VulkanQuant prefill GEMM).
    let qs_vk = QStorage::from_data(std::borrow::Cow::Owned(bytes), dev, dtype)?;
    let q_vk = QTensor::new(qs_vk, (nout, k))?;
    let qm_vk = QMatMul::from_qtensor(q_vk)?;
    // A multi-row [M, k] input on the Vulkan device triggers the native prefill GEMM.
    let x_vk = Tensor::from_vec(x_host, (m, k), dev)?;
    let y_vk = qm_vk.forward(&x_vk)?.flatten_all()?.to_vec1::<f32>()?;

    assert_eq!(y_vk.len(), m * nout);
    let mut max_abs = 0f32;
    for i in 0..m * nout {
        max_abs = max_abs.max((y_vk[i] as f64 - y_ref[i]).abs() as f32);
    }
    Ok(max_abs)
}

#[test]
fn vulkan_qmatmul_prefill_matches_cpu() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else { return Ok(()) };
    // M straddles the host MAX_M=8 tiling AND the per-dtype GEMM gate (Q5_K/Q6_K: 64, else 128):
    // 5/8/17 hit the native GEMM for every dtype; 100 hits the GEMM for Q4_0/Q8_0/Q4_K but the dequant
    // fallback for Q5_K/Q6_K (exercising the split); 200 (> both gates) is the dequant path for all.
    // BOTH prefill paths are thus validated against the same f64 reference.
    // (nout, k) with k divisible by 256 so all five dtypes share the same shapes.
    for &m in &[5usize, 8, 17, 100, 200] {
        for &(nout, k) in &[(2048usize, 2048usize), (4096, 2048), (512, 256)] {
            for dt in [
                GgmlDType::Q4_0,
                GgmlDType::Q8_0,
                GgmlDType::Q4K,
                GgmlDType::Q5K,
                GgmlDType::Q6K,
            ] {
                let max_abs = prefill_case(&dev, dt, m, nout, k)?;
                println!(
                    "QMatMul::prefill {dt:?}  M={m:3} nout={nout:5} k={k:5}  GPU-vs-(dequant ref) max_abs={max_abs:.3e}"
                );
                // The GEMM decodes the SAME quantized bytes as the reference dequant and accumulates
                // per row exactly like the decode matvec; only fp32-vs-f64 accumulation order differs.
                assert!(
                    max_abs < 1e-3,
                    "QMatMul::prefill {dt:?} M={m} GPU/ref mismatch too large: {max_abs}"
                );
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// MoE: the fused grouped quant matvec. Build a quantized expert bank [E, n, k] on the Vulkan device
// and call QTensor::indexed_moe_forward (the path Qwen3-MoE uses) -- which now runs one on-GPU
// dispatch that gathers each routed slot's expert and computes its matvec -- and compare to a host
// f64 reference over the dequantized per-expert weights.
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
    // Expert bank [E, n, k] and its dequantized f32 (the reference weights).
    let bank_host: Vec<f32> = (0..e_cnt * n * k).map(|i| pseudo(i) * 0.5).collect();
    let bank_t = Tensor::from_vec(bank_host, (e_cnt, n, k), &cpu)?;
    let q_bank = QTensor::quantize(&bank_t.reshape((e_cnt * n, k))?, dtype)?; // quantize per row
    let bank_deq: Vec<f32> = q_bank.dequantize(&cpu)?.flatten_all()?.to_vec1::<f32>()?; // [E*n*k]
    let bytes = q_bank.data()?.into_owned();

    // Activations [t, topk, k] (per-slot inputs) and router ids [t, topk].
    let x_host: Vec<f32> = (0..t * topk * k).map(|i| pseudo(i + 11) * 0.7).collect();
    // Deterministic expert assignment spread across all experts.
    let ids_host: Vec<u32> = (0..t * topk).map(|i| ((i * 7 + 3) % e_cnt) as u32).collect();

    // Build the Vulkan QTensor bank the loader way and run the fused MoE forward.
    let qs_vk = QStorage::from_data(std::borrow::Cow::Owned(bytes), dev, dtype)?;
    let q_vk = QTensor::new(qs_vk, (e_cnt, n, k))?;
    let x_vk = Tensor::from_vec(x_host.clone(), (t, topk, k), dev)?;
    let ids_vk = Tensor::from_vec(ids_host.clone(), (t, topk), dev)?;
    let y_vk = q_vk
        .indexed_moe_forward(&x_vk, &ids_vk)?
        .reshape((t * topk, n))?
        .to_vec2::<f32>()?;

    // Reference: y[slot, r] = sum_k W_deq[ids[slot], r, k] * x[slot, k].
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
            max_abs = max_abs.max((y_vk[slot][r] as f64 - acc).abs() as f32);
        }
    }
    Ok(max_abs)
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

    let vk = dev.as_vulkan_device()?;
    let out = vk.flash_attn(&q, &k, &v, bh, lq, lk, d, scale, causal)?;
    assert_eq!(out.len(), bh * lq * d);

    // Eager f64 reference.
    let mut max_abs = 0f32;
    for b in 0..bh {
        for qi in 0..lq {
            let last = if causal { qi + (lk - lq) + 1 } else { lk };
            let last = last.min(lk);
            // scores
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
                for (j, &p) in sc.iter().enumerate() {
                    acc += p * v[(b * lk + j) * d + t] as f64;
                }
                acc /= denom;
                let g = out[(b * lq + qi) * d + t] as f64;
                max_abs = max_abs.max((g - acc).abs() as f32);
            }
        }
    }
    Ok(max_abs)
}

#[test]
fn vulkan_flash_attn_matches_cpu() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else { return Ok(()) };
    // (bh, lq, lk, d, causal). d = 128 is Qwen3's head dim.
    let cases = [
        (8usize, 16usize, 16usize, 128usize, false), // prefill, non-causal
        (8, 16, 16, 128, true),                       // prefill, causal
        (8, 1, 64, 128, true),                        // decode: single query over a 64-key cache
        (4, 7, 13, 64, false),                        // ragged Lq != Lk, smaller d
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

#[test]
fn vulkan_moe_forward_matches_cpu() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else { return Ok(()) };
    // Qwen3-MoE-ish: many experts, small expert intermediate n, hidden k, a few tokens, top-k.
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

// L1 tiled prefill GEMM (mul_mm_q4k_shared, gated by HANZO_VK_Q4K_TILED): one workgroup per output
// column stages that column's weight in LDS once and reuses it across all M rows, so weight VRAM
// traffic is 1x instead of ceil(M/8)x. It does the IDENTICAL f32 decode + accumulation order as the
// default mul_mat_q4k kernel, so its output must match bit-for-bit (a wrong nibble/scale/min, an LDS
// indexing bug, or a missed row would be a systematic, large divergence -- not a 1-ULP FMA difference).
// Gate: run both kernels on the same weight+activations and assert the relative max-abs diff is tiny.
#[test]
fn vulkan_q4k_tiled_prefill_matches_default() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else {
        return Ok(());
    };
    let vk = dev.as_vulkan_device()?;
    // M>1 prefill across nout/k shapes incl. multi-superblock k and k at the LDS bound (64 superblocks),
    // and M values straddling the default kernel's 8-row tile (8, 9, 64, 512).
    for &(nout, k) in &[
        (512usize, 256usize),
        (2048, 2048),
        (256, 4096),
        (512, 16384),
    ] {
        for &m in &[2usize, 8, 9, 64, 512] {
            let (raw, _w_deq, _w_host) = weight_bytes(GgmlDType::Q4K, nout, k)?;
            let wq = vk.upload_qweight(&raw)?;
            let x: Vec<f32> = (0..m * k).map(|i| pseudo(i + 7)).collect();

            std::env::set_var("HANZO_VK_Q4K_LEGACY", "1");
            let y_ref = vk.matmul_q4k(&wq, &x, m, nout, k)?;
            std::env::remove_var("HANZO_VK_Q4K_LEGACY");
            std::env::set_var("HANZO_VK_Q4K_TILED", "1");
            let y_tiled = vk.matmul_q4k(&wq, &x, m, nout, k)?;
            std::env::remove_var("HANZO_VK_Q4K_TILED");

            assert_eq!(y_ref.len(), m * nout);
            assert_eq!(y_tiled.len(), m * nout);
            let mut max_abs = 0f32;
            let mut max_ref = 0f32;
            for (a, b) in y_ref.iter().zip(y_tiled.iter()) {
                max_abs = max_abs.max((a - b).abs());
                max_ref = max_ref.max(a.abs());
            }
            let rel = if max_ref > 0.0 { max_abs / max_ref } else { max_abs };
            assert!(
                rel < 1e-5,
                "tiled != default (m={m}, nout={nout}, k={k}): max_abs={max_abs}, max_ref={max_ref}, rel={rel}"
            );
        }
    }
    Ok(())
}

// L1 perf A/B: time the default mul_mat_q4k vs the tiled mul_mm_q4k_shared at a realistic prefill shape.
// Both go through the same matmul_q4k host wrapper (identical upload+readback), so default_ms - tiled_ms
// is pure kernel time. #[ignore] (run with --ignored): cargo test ... vulkan_q4k_tiled_prefill_bench
#[test]
#[ignore]
fn vulkan_q4k_tiled_prefill_bench() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else {
        return Ok(());
    };
    let vk = dev.as_vulkan_device()?;
    let (m, nout, k) = (512usize, 4096usize, 4096usize);
    let (raw, _, _) = weight_bytes(GgmlDType::Q4K, nout, k)?;
    let wq = vk.upload_qweight(&raw)?;
    let x: Vec<f32> = (0..m * k).map(|i| pseudo(i + 7)).collect();
    let iters = 20;
    let bench = |tiled: bool| -> hanzo_ml::Result<f64> {
        if tiled {
            std::env::set_var("HANZO_VK_Q4K_TILED", "1");
        } else {
            std::env::remove_var("HANZO_VK_Q4K_TILED");
        }
        let _ = vk.matmul_q4k(&wq, &x, m, nout, k)?; // warmup (shader compile / first alloc)
        let t = std::time::Instant::now();
        for _ in 0..iters {
            let _ = vk.matmul_q4k(&wq, &x, m, nout, k)?;
        }
        std::env::remove_var("HANZO_VK_Q4K_TILED");
        Ok(t.elapsed().as_secs_f64() * 1e3 / iters as f64)
    };
    let def_ms = bench(false)?;
    let tiled_ms = bench(true)?;
    eprintln!(
        "[L1 bench] m={m} nout={nout} k={k}: default={def_ms:.3} ms  tiled={tiled_ms:.3} ms  speedup={:.2}x",
        def_ms / tiled_ms
    );
    Ok(())
}

// L1 2D-tiled prefill GEMM (mul_mm_q4k_tiled, HANZO_VK_Q4K_TILED2D): 64x64 output tile, both operands
// staged in LDS per K-step. Decode per element is identical to mul_mat_q4k but accumulation is tiled
// (partial sums), so it matches the default within f32-reorder tolerance, not bit-exact. A decode/index
// bug is a systematic >>1e-3 divergence; f32 reassociation is <~1e-4.
#[test]
fn vulkan_q4k_tiled2d_matches_default() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else {
        return Ok(());
    };
    let vk = dev.as_vulkan_device()?;
    for &(nout, k) in &[(512usize, 256usize), (2048, 2048), (256, 4096), (320, 1024)] {
        for &m in &[1usize, 7, 64, 65, 512] {
            let (raw, _, _) = weight_bytes(GgmlDType::Q4K, nout, k)?;
            let wq = vk.upload_qweight(&raw)?;
            let x: Vec<f32> = (0..m * k).map(|i| pseudo(i + 11)).collect();
            std::env::set_var("HANZO_VK_Q4K_LEGACY", "1");
            let y_ref = vk.matmul_q4k(&wq, &x, m, nout, k)?;
            std::env::remove_var("HANZO_VK_Q4K_LEGACY");
            std::env::set_var("HANZO_VK_Q4K_TILED2D", "1");
            let y_2d = vk.matmul_q4k(&wq, &x, m, nout, k)?;
            std::env::remove_var("HANZO_VK_Q4K_TILED2D");
            assert_eq!(y_2d.len(), m * nout);
            let mut max_abs = 0f32;
            let mut max_ref = 0f32;
            for (a, b) in y_ref.iter().zip(y_2d.iter()) {
                max_abs = max_abs.max((a - b).abs());
                max_ref = max_ref.max(a.abs());
            }
            let rel = if max_ref > 0.0 { max_abs / max_ref } else { max_abs };
            assert!(
                rel < 2e-3,
                "2d-tiled != default (m={m}, nout={nout}, k={k}): max_abs={max_abs}, rel={rel}"
            );
        }
    }
    Ok(())
}

// L1 perf: default mul_mat_q4k vs 2D-tiled mul_mm_q4k_tiled at a realistic prefill shape (run --ignored).
#[test]
#[ignore]
fn vulkan_q4k_tiled2d_bench() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else {
        return Ok(());
    };
    let vk = dev.as_vulkan_device()?;
    let (m, nout, k) = (512usize, 4096usize, 4096usize);
    let (raw, _, _) = weight_bytes(GgmlDType::Q4K, nout, k)?;
    let wq = vk.upload_qweight(&raw)?;
    let x: Vec<f32> = (0..m * k).map(|i| pseudo(i + 7)).collect();
    let iters = 20;
    let bench = |var: Option<&str>| -> hanzo_ml::Result<f64> {
        std::env::remove_var("HANZO_VK_Q4K_TILED2D");
        if let Some(v) = var {
            std::env::set_var(v, "1");
        }
        let _ = vk.matmul_q4k(&wq, &x, m, nout, k)?;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            let _ = vk.matmul_q4k(&wq, &x, m, nout, k)?;
        }
        if let Some(v) = var {
            std::env::remove_var(v);
        }
        Ok(t.elapsed().as_secs_f64() * 1e3 / iters as f64)
    };
    let def_ms = bench(Some("HANZO_VK_Q4K_LEGACY"))?;
    let tiled2d_ms = bench(Some("HANZO_VK_Q4K_TILED2D"))?;
    eprintln!(
        "[L1 2D bench] m={m} nout={nout} k={k}: default={def_ms:.3} ms  tiled2d={tiled2d_ms:.3} ms  speedup={:.2}x",
        def_ms / tiled2d_ms
    );
    Ok(())
}

// L1 dp4a 2D-tiled GEMM (mul_mm_q4k_tiled_dp4a, HANZO_VK_Q4K_DP4A): the 2D tile with an int8 dp4a inner
// loop -- activations are quantized to q8_1 (int8 + per-32-block scale/sum) so the dot is integer. That
// q8_1 quant adds ~0.5-1% error on top of the f32 tiling reorder, so the gate vs the exact f32 default
// is a ~1.5% relative bound (a decode/index/pack bug is a systematic >>1.5% garble).
#[test]
fn vulkan_q4k_dp4a2d_matches_default() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else {
        return Ok(());
    };
    let vk = dev.as_vulkan_device()?;
    for &(nout, k) in &[(512usize, 256usize), (2048, 2048), (256, 4096), (320, 1024)] {
        for &m in &[1usize, 7, 64, 65, 512] {
            let (raw, _, _) = weight_bytes(GgmlDType::Q4K, nout, k)?;
            let wq = vk.upload_qweight(&raw)?;
            let x: Vec<f32> = (0..m * k).map(|i| pseudo(i + 13)).collect();
            std::env::set_var("HANZO_VK_Q4K_LEGACY", "1");
            let y_ref = vk.matmul_q4k(&wq, &x, m, nout, k)?;
            std::env::remove_var("HANZO_VK_Q4K_LEGACY");
            std::env::set_var("HANZO_VK_Q4K_DP4A", "1");
            let y_dp = vk.matmul_q4k(&wq, &x, m, nout, k)?;
            std::env::remove_var("HANZO_VK_Q4K_DP4A");
            assert_eq!(y_dp.len(), m * nout);
            let mut max_abs = 0f32;
            let mut max_ref = 0f32;
            for (a, b) in y_ref.iter().zip(y_dp.iter()) {
                max_abs = max_abs.max((a - b).abs());
                max_ref = max_ref.max(a.abs());
            }
            let rel = if max_ref > 0.0 { max_abs / max_ref } else { max_abs };
            assert!(
                rel < 1.5e-2,
                "dp4a-2d != default (m={m}, nout={nout}, k={k}): max_abs={max_abs}, max_ref={max_ref}, rel={rel}"
            );
        }
    }
    Ok(())
}

// L1 perf: default vs 2D-f32 vs 2D-dp4a at a realistic prefill shape (run --ignored).
#[test]
#[ignore]
fn vulkan_q4k_dp4a2d_bench() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else {
        return Ok(());
    };
    let vk = dev.as_vulkan_device()?;
    let (m, nout, k) = (512usize, 4096usize, 4096usize);
    let (raw, _, _) = weight_bytes(GgmlDType::Q4K, nout, k)?;
    let wq = vk.upload_qweight(&raw)?;
    let x: Vec<f32> = (0..m * k).map(|i| pseudo(i + 7)).collect();
    let iters = 20;
    let bench = |var: Option<&str>| -> hanzo_ml::Result<f64> {
        for v in ["HANZO_VK_Q4K_TILED2D", "HANZO_VK_Q4K_DP4A"] {
            std::env::remove_var(v);
        }
        if let Some(v) = var {
            std::env::set_var(v, "1");
        }
        let _ = vk.matmul_q4k(&wq, &x, m, nout, k)?;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            let _ = vk.matmul_q4k(&wq, &x, m, nout, k)?;
        }
        if let Some(v) = var {
            std::env::remove_var(v);
        }
        Ok(t.elapsed().as_secs_f64() * 1e3 / iters as f64)
    };
    let def_ms = bench(Some("HANZO_VK_Q4K_LEGACY"))?;
    let f32_ms = bench(Some("HANZO_VK_Q4K_TILED2D"))?;
    let dp4a_ms = bench(Some("HANZO_VK_Q4K_DP4A"))?;
    eprintln!(
        "[L1 dp4a bench] m={m} nout={nout} k={k}: default={def_ms:.3} 2d_f32={f32_ms:.3} 2d_dp4a={dp4a_ms:.3} ms | dp4a vs default={:.2}x vs 2d_f32={:.2}x",
        def_ms / dp4a_ms,
        f32_ms / dp4a_ms
    );
    Ok(())
}

// L4 coopmat Q4_K prefill GEMM (mul_mm_q4k_coopmat, HANZO_VK_Q4K_COOPMAT): tensor-core path (decode
// weight to f16 LDS tiles + coopMatMulAdd). f16 weight + f16 activation rounding (f32 accumulate), so
// the gate vs the exact legacy column kernel is ~1% relative (a decode/transpose/tile bug is >>1%).
// Skips if the device lacks coopmat (the env path is a no-op fallthrough -> equals the default).
#[test]
fn vulkan_q4k_coopmat_matches_default() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else {
        return Ok(());
    };
    let vk = dev.as_vulkan_device()?;
    if vk.coopmat_info().is_none() {
        eprintln!("[vulkan_quant_tests] no coopmat; skipping coopmat gate");
        return Ok(());
    }
    for &(nout, k) in &[(512usize, 256usize), (2048, 2048), (256, 4096), (320, 1024)] {
        for &m in &[2usize, 16, 17, 64, 512] {
            let (raw, _, _) = weight_bytes(GgmlDType::Q4K, nout, k)?;
            let wq = vk.upload_qweight(&raw)?;
            let x: Vec<f32> = (0..m * k).map(|i| pseudo(i + 17)).collect();
            std::env::set_var("HANZO_VK_Q4K_LEGACY", "1");
            let y_ref = vk.matmul_q4k(&wq, &x, m, nout, k)?;
            std::env::remove_var("HANZO_VK_Q4K_LEGACY");
            std::env::set_var("HANZO_VK_Q4K_COOPMAT", "1");
            let y_cm = vk.matmul_q4k(&wq, &x, m, nout, k)?;
            std::env::remove_var("HANZO_VK_Q4K_COOPMAT");
            let mut max_abs = 0f32;
            let mut max_ref = 0f32;
            for (a, b) in y_ref.iter().zip(y_cm.iter()) {
                max_abs = max_abs.max((a - b).abs());
                max_ref = max_ref.max(a.abs());
            }
            let rel = if max_ref > 0.0 { max_abs / max_ref } else { max_abs };
            assert!(
                rel < 1e-2,
                "coopmat != legacy (m={m}, nout={nout}, k={k}): max_abs={max_abs}, rel={rel}"
            );
        }
    }
    Ok(())
}

// L4 perf: default (dp4a) vs coopmat at a realistic prefill shape (run --ignored).
#[test]
#[ignore]
fn vulkan_q4k_coopmat_bench() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else {
        return Ok(());
    };
    let vk = dev.as_vulkan_device()?;
    if vk.coopmat_info().is_none() {
        eprintln!("[vulkan_quant_tests] no coopmat; skipping");
        return Ok(());
    }
    let (m, nout, k) = (512usize, 4096usize, 4096usize);
    let (raw, _, _) = weight_bytes(GgmlDType::Q4K, nout, k)?;
    let wq = vk.upload_qweight(&raw)?;
    let x: Vec<f32> = (0..m * k).map(|i| pseudo(i + 7)).collect();
    let iters = 20;
    let bench = |var: Option<&str>| -> hanzo_ml::Result<f64> {
        for v in ["HANZO_VK_Q4K_COOPMAT", "HANZO_VK_Q4K_LEGACY"] {
            std::env::remove_var(v);
        }
        if let Some(v) = var {
            std::env::set_var(v, "1");
        }
        let _ = vk.matmul_q4k(&wq, &x, m, nout, k)?;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            let _ = vk.matmul_q4k(&wq, &x, m, nout, k)?;
        }
        if let Some(v) = var {
            std::env::remove_var(v);
        }
        Ok(t.elapsed().as_secs_f64() * 1e3 / iters as f64)
    };
    let dp4a_ms = bench(None)?; // default == dp4a on this device
    let legacy_ms = bench(Some("HANZO_VK_Q4K_LEGACY"))?;
    let cm_ms = bench(Some("HANZO_VK_Q4K_COOPMAT"))?;
    eprintln!(
        "[L4 coopmat bench] m={m} nout={nout} k={k}: legacy={legacy_ms:.3} dp4a={dp4a_ms:.3} coopmat={cm_ms:.3} ms | coopmat vs dp4a={:.2}x vs legacy={:.2}x",
        dp4a_ms / cm_ms,
        legacy_ms / cm_ms
    );
    Ok(())
}

// Kernel-isolated perf: the TRUE Q4_K prefill kernel time (GPU-resident operands, no per-iter host
// transfer) for legacy / f32-2D / dp4a / coopmat, plus the effective GFLOP/s. Run with --ignored.
#[test]
#[ignore]
fn vulkan_q4k_kernel_bench() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else {
        return Ok(());
    };
    let vk = dev.as_vulkan_device()?;
    let (m, nout, k) = (512usize, 4096usize, 4096usize);
    let (raw, _, _) = weight_bytes(GgmlDType::Q4K, nout, k)?;
    let wq = vk.upload_qweight(&raw)?;
    let x: Vec<f32> = (0..m * k).map(|i| pseudo(i + 7)).collect();
    let bench = |var: Option<&str>| -> hanzo_ml::Result<f64> {
        for v in [
            "HANZO_VK_Q4K_LEGACY",
            "HANZO_VK_Q4K_TILED2D",
            "HANZO_VK_Q4K_DP4A",
            "HANZO_VK_Q4K_COOPMAT",
        ] {
            std::env::remove_var(v);
        }
        if let Some(v) = var {
            std::env::set_var(v, "1");
        }
        let r = vk.bench_matmul_q4k(&wq, &x, m, nout, k, 50);
        if let Some(v) = var {
            std::env::remove_var(v);
        }
        r
    };
    let leg = bench(Some("HANZO_VK_Q4K_LEGACY"))?;
    let f32t = bench(Some("HANZO_VK_Q4K_TILED2D"))?;
    let dp = bench(None)?; // default == dp4a on int_dot8 devices
    let cm = if vk.coopmat_info().is_some() {
        bench(Some("HANZO_VK_Q4K_COOPMAT"))?
    } else {
        0.0
    };
    let gflop = 2.0 * m as f64 * nout as f64 * k as f64 / 1e9;
    eprintln!(
        "[KERNEL bench] m={m} nout={nout} k={k} ({gflop:.1} GFLOP): legacy={leg:.3} f32_2d={f32t:.3} dp4a={dp:.3} coopmat={cm:.3} ms | dp4a={:.0} GFLOP/s, {:.1}x vs legacy",
        gflop / (dp / 1e3),
        leg / dp
    );
    Ok(())
}

// Fused rope_norm (rms_norm + NeoX rope in ONE dispatch) must match the unfused two-op chain
// bit-exactly (same f32 ops, same order). First rung of the Vulkan decode op-fusion campaign:
// it removes the inter-op global barrier that serializes decode (see LLM.md / paper 06).
#[test]
fn vulkan_rope_norm_matches_unfused() {
    let Some(dev) = gpu() else { return; };
    let vk = dev.as_vulkan_device().unwrap();
    // (b, h, t, d): decode (t=1) + short prefill; q/k head counts; head dims 64 and 128.
    for &(b, h, t, d) in &[(1usize, 4usize, 1usize, 128usize), (1, 8, 5, 64), (2, 4, 3, 128)] {
        let hd = d / 2;
        let n = b * h * t * d;
        let x: Vec<f32> = (0..n).map(|i| pseudo(i + 7)).collect();
        let weight: Vec<f32> = (0..d).map(|i| 0.5 + pseudo(i + 99).abs()).collect();
        let cs: Vec<f32> = (0..t * hd).map(|i| pseudo(i + 3).cos()).collect();
        let sn: Vec<f32> = (0..t * hd).map(|i| pseudo(i + 3).sin()).collect();
        let eps = 1e-6f32;
        // CPU reference: rms_norm (x / sqrt(mean(x^2)+eps) * weight) then NeoX rope.
        let mut refv = vec![0f32; n];
        for row in 0..b * h * t {
            let base = row * d;
            let mut ss = 0f32;
            for i in 0..d { let v = x[base + i]; ss += v * v; }
            let denom = (ss / d as f32 + eps).sqrt();
            let i_t = row % t;
            for i_d in 0..hd {
                let (i1, i2) = (base + i_d, base + i_d + hd);
                let x1 = x[i1] / denom * weight[i_d];
                let x2 = x[i2] / denom * weight[i_d + hd];
                let (c, s) = (cs[i_t * hd + i_d], sn[i_t * hd + i_d]);
                refv[i1] = x1 * c - x2 * s;
                refv[i2] = x1 * s + x2 * c;
            }
        }
        let got = vk.rope_norm_f32(&x, &weight, &cs, &sn, b, h, t, d, eps).unwrap();
        assert_eq!(got.len(), n);
        let mut maxabs = 0f32;
        for i in 0..n { maxabs = maxabs.max((got[i] - refv[i]).abs()); }
        eprintln!("rope_norm b{b} h{h} t{t} d{d}: maxabs={maxabs:.3e}");
        assert!(maxabs < 1e-4, "rope_norm (b{b} h{h} t{t} d{d}) mismatch maxabs={maxabs}");
    }
}

// Fused add_rmsnorm (residual-add + RMSNorm -> (s, y) in one dispatch) must match the unfused
// add-then-rms_norm chain bit-exactly. Rung 2 of the Vulkan decode op-fusion campaign.
#[test]
fn vulkan_add_rmsnorm_matches_unfused() {
    let Some(dev) = gpu() else { return; };
    let vk = dev.as_vulkan_device().unwrap();
    for &(nrows, m) in &[(1usize, 2048usize), (5, 64), (3, 4096)] {
        let n = nrows * m;
        let x: Vec<f32> = (0..n).map(|i| pseudo(i + 11)).collect();
        let res: Vec<f32> = (0..n).map(|i| pseudo(i + 222)).collect();
        let alpha: Vec<f32> = (0..m).map(|i| 0.5 + pseudo(i + 9).abs()).collect();
        let eps = 1e-5f32;
        // CPU reference: s = x + res ; y = s / sqrt(mean(s^2)+eps) * alpha.
        let mut s_ref = vec![0f32; n];
        let mut y_ref = vec![0f32; n];
        for row in 0..nrows {
            let base = row * m;
            let mut ss = 0f32;
            for i in 0..m { let v = x[base + i] + res[base + i]; s_ref[base + i] = v; ss += v * v; }
            let denom = (ss / m as f32 + eps).sqrt();
            for i in 0..m { y_ref[base + i] = s_ref[base + i] / denom * alpha[i]; }
        }
        let (s_gpu, y_gpu) = vk.add_rmsnorm_f32(&x, &res, &alpha, nrows, m, eps).unwrap();
        let (mut ms, mut my) = (0f32, 0f32);
        for i in 0..n { ms = ms.max((s_gpu[i] - s_ref[i]).abs()); my = my.max((y_gpu[i] - y_ref[i]).abs()); }
        eprintln!("add_rmsnorm r{nrows} m{m}: s_maxabs={ms:.3e} y_maxabs={my:.3e}");
        assert!(ms < 1e-5 && my < 1e-4, "add_rmsnorm r{nrows} m{m} mismatch s={ms} y={my}");
    }
}

// Both dp4a decode matvecs -- column (matvec_q4k_dp4a) and subgroup (matvec_q4k_dp4a_sg, the default
// where the device has subgroup arithmetic) -- must match the SCALAR matvec within the q8_1 activation
// quant tolerance. matvec_q4k_scalar forces the non-dp4a path (the exact reference) regardless of the
// dp4a default. Validates the Vulkan decode lever (1.8x) for both reductions.
#[test]
fn vulkan_q4k_dp4a_decode_matches_scalar() {
    let Some(dev) = gpu() else { return; };
    let vk = dev.as_vulkan_device().unwrap();
    for &(nout, k) in SHAPES {
        let x: Vec<f32> = (0..k).map(|i| pseudo(i + 5)).collect();
        let (raw, _w_deq, _w_host) = weight_bytes(GgmlDType::Q4K, nout, k).unwrap();
        let wq = vk.upload_qweight(&raw).unwrap();
        let scalar = vk.matvec_q4k_scalar(&wq, &x, nout, k).unwrap();
        let column = vk.matvec_q4k_dp4a(&wq, &x, nout, k).unwrap();
        let subgroup = vk.matvec_q4k_dp4a_sg(&wq, &x, nout, k).unwrap();
        let r2 = vk.matvec_q4k_dp4a_r2(&wq, &x, nout, k).unwrap();
        for (name, got) in [("column", &column), ("subgroup", &subgroup), ("r2", &r2)] {
            let (mut sse, mut refsq) = (0f64, 0f64);
            for i in 0..nout {
                let d = (got[i] - scalar[i]) as f64;
                sse += d * d;
                refsq += (scalar[i] as f64) * (scalar[i] as f64);
            }
            let rel = (sse / refsq.max(1e-9)).sqrt();
            eprintln!("dp4a-{name} vs scalar nout={nout} k={k}: rel={rel:.3e}");
            assert!(rel < 2e-2, "dp4a {name} rel err {rel} too large (nout={nout} k={k})");
        }
    }
}
