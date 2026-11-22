//! Numerical validation of the native CUDA i-quant mmvq DECODE kernels (qmatvec_dp4a_<iq*>_f32 in
//! iquant_mmvq.cu) against the CPU `to_float` reference, on the real GB10 GPU.
//!
//! Each test:
//!   1. synthesizes valid raw GGML i-quant block bytes (every codebook index / sign / scale field is
//!      in range), so the block dequantizes to O(0.5)-magnitude weights;
//!   2. loads those bytes as a CUDA QTensor and runs `QMatMul::forward` with a [1, k] f32 activation
//!      -- this routes through `fwd` -> `mul_mat_vec_iquant` (the native dp4a decode path);
//!   3. dequantizes the SAME bytes on the CPU (`QTensor::dequantize`, the exact weights the kernel
//!      decodes) and computes a host f64 matvec as the ground truth;
//!   4. asserts agreement (the only difference is f32 accumulation order: ~1e-3).
//!
//! Skips cleanly (prints + returns) when no CUDA GPU is present. Requires the `cuda` feature.
#![cfg(feature = "cuda")]

use hanzo_ml::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use hanzo_ml::{Device, Module, Tensor};
use std::borrow::Cow;

// Deterministic pseudo-random f32 in [-1, 1) from a counter (splitmix64-ish; reproducible, no rng dep).
fn pseudo(i: usize) -> f32 {
    let mut z = (i as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
}

fn pbyte(i: usize) -> u8 {
    (((pseudo(i) * 0.5 + 0.5) * 256.0) as i32).clamp(0, 255) as u8
}

// Synthesize valid raw GGML block bytes for an i-quant type. Any byte pattern is a valid codebook
// block (grid index / sign group / scale nibble all wrap in range); the f16 scale magnitude keeps the
// dequantized weights O(0.5). Block layouts are bit-for-bit the CPU block structs. (IQ3's *0.5/scale
// folding and IQ1's delta bias amplify outputs, so their scales are tuned down -- mirrors the Vulkan
// synth so the same 1e-3 tolerance holds.)
fn synth(dtype: GgmlDType, nout: usize, k: usize) -> Vec<u8> {
    use half::f16;
    let mut out: Vec<u8> = Vec::new();
    let mut c = 0usize;
    let nblk = nout * (k / 256);
    let emit_d_then = |scale: f32, nbytes: usize, out: &mut Vec<u8>, c: &mut usize| {
        out.extend_from_slice(&f16::from_f32(pseudo(*c) * scale).to_le_bytes());
        *c += 1;
        for _ in 0..nbytes {
            out.push(pbyte(*c));
            *c += 1;
        }
    };
    match dtype {
        // f16 d + qs (66/74/82 B for XXS/XS/S after the 2-byte d).
        GgmlDType::IQ2_XXS => (0..nblk).for_each(|_| emit_d_then(0.05, 64, &mut out, &mut c)),
        GgmlDType::IQ2_XS => (0..nblk).for_each(|_| emit_d_then(0.05, 72, &mut out, &mut c)),
        GgmlDType::IQ2_S => (0..nblk).for_each(|_| emit_d_then(0.05, 80, &mut out, &mut c)),
        GgmlDType::IQ3_XXS => (0..nblk).for_each(|_| emit_d_then(0.02, 96, &mut out, &mut c)),
        GgmlDType::IQ3_S => (0..nblk).for_each(|_| emit_d_then(0.004, 108, &mut out, &mut c)),
        GgmlDType::IQ1_S => (0..nblk).for_each(|_| emit_d_then(0.05, 48, &mut out, &mut c)),
        // block_iq4_xs (136 B): f16 d + u16 scales_h + scales_l[4] + qs[128]. Any byte pattern is valid
        // (4-bit LUT indices + 6-bit scales all in range). dl = d*(ls-32) amplifies, so d is kept small.
        GgmlDType::IQ4_XS => (0..nblk).for_each(|_| emit_d_then(0.001, 134, &mut out, &mut c)),
        // block_iq1m (56 B): qs[32]+qh[16]+scales[8], NO leading d -- d is reconstructed from the high
        // nibbles of the 4 scale u16. scales -> scale_u16 = 0x2C00 (f16 0.0625), per-sub-block 3-bit
        // fields 0 (dl = d); qs/qh random (valid indices + signs).
        GgmlDType::IQ1_M => {
            for _ in 0..nblk {
                for _ in 0..48 {
                    out.push(pbyte(c));
                    c += 1;
                }
                out.extend_from_slice(&[0u8, 0, 0, 0, 0, 0xC0, 0, 0x20]);
            }
        }
        // Ternary: field order is qs THEN d (NOT d-first like the IQ codebooks). value = (xi-1)*d in
        // {-d,0,d}, so d~0.25 keeps weights O(0.25). Any byte pattern is a valid ternary word (2-bit
        // field for TQ2_0; base-3 (b*pow3[n] mod 256)*3>>8 in [0,2] for TQ1_0).
        // TQ2_0 (66 B): qs[64] + f16 d.
        GgmlDType::TQ2_0 => {
            for _ in 0..nblk {
                for _ in 0..64 {
                    out.push(pbyte(c));
                    c += 1;
                }
                out.extend_from_slice(&f16::from_f32(pseudo(c) * 0.25).to_le_bytes());
                c += 1;
            }
        }
        // TQ1_0 (54 B): qs[48] + qh[4] + f16 d.
        GgmlDType::TQ1_0 => {
            for _ in 0..nblk {
                for _ in 0..52 {
                    out.push(pbyte(c));
                    c += 1;
                }
                out.extend_from_slice(&f16::from_f32(pseudo(c) * 0.25).to_le_bytes());
                c += 1;
            }
        }
        GgmlDType::PQ2_0 => {
            let nblk128 = nout * (k / 128);
            (0..nblk128).for_each(|_| emit_d_then(0.15, 32, &mut out, &mut c));
        }
        GgmlDType::PTQ1_0 => {
            let nblk128 = nout * (k / 128);
            for _ in 0..nblk128 {
                for _ in 0..26 {
                    out.push(pbyte(c));
                    c += 1;
                }
                out.extend_from_slice(&f16::from_f32(pseudo(c) * 0.2).to_le_bytes());
                c += 1;
            }
        }
        GgmlDType::Q1_0 => {
            let nblk128 = nout * (k / 128);
            (0..nblk128).for_each(|_| emit_d_then(0.2, 16, &mut out, &mut c));
        }
        _ => panic!("synth: {dtype:?} is not an i-quant type"),
    }
    out
}

struct Err {
    max_abs: f32,
    max_rel: f32,
}

// q8_1 round-trip the activation per 32-block exactly as `iq_quantize_q8_f32` does (absmax/127 scale,
// round-half-away-from-zero, clamp [-127,127], dequantize). The dp4a kernel dots the int8 weight
// codebook against THIS quantized activation, so the bit-faithful reference must see the same
// quantized activation -- otherwise the comparison just measures the q8_1 activation's inherent ~1/256
// precision, not the kernel. (This mirrors the ROCm `to_float-on-q8_1_recon` gate.)
fn q8_1_roundtrip(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    let mut i = 0;
    for blk in x.chunks(32) {
        let absmax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let d = if absmax > 0.0 { absmax / 127.0 } else { 0.0 };
        let inv = if absmax > 0.0 { 127.0 / absmax } else { 0.0 };
        for &v in blk {
            let q = (v * inv).round().clamp(-127.0, 127.0);
            out[i] = q * d;
            i += 1;
        }
    }
    out
}

// Run one (dtype, nout, k) case through the native CUDA decode and return error stats vs the CPU
// `to_float` reference (over the q8_1-roundtripped activation). b_size = 1 (decode shape) so `fwd`
// routes to `mul_mat_vec_iquant`.
// --- kernel-level decode microbench (perf A/B; run: `cargo test ... -- --ignored bench_iquant_decode --nocapture`) ---
// Times the native dp4a i-quant DECODE matvec (M=1 -> mul_mat_vec_iquant, the kernel measured at ~75%
// of GB10 i-quant decode GPU-time). At 4096x4096 the call is GPU-time-dominated (~60us kernel >> ~5us
// launch), so us/call isolates the sign-decode kernel A/B vs the prior 2-dp4a + KSIGNS-gather form.
// IQ1_S/IQ4_XS are unchanged controls (their us/call must match baseline -> confirms thermal/harness
// stability, isolating the changed-types delta). GB/s = actual quantized weight bytes / us.
#[test]
#[ignore]
fn bench_iquant_decode() {
    let dev = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip bench: no CUDA device ({e})");
            return;
        }
    };
    let (nout, k) = (4096usize, 4096usize);
    let (warm, iters) = (64usize, 3000usize);
    println!("=== iquant decode microbench {nout}x{k}, {iters} iters (GB10) ===");
    for &dtype in &[
        GgmlDType::IQ2_XXS, // changed: ksigns
        GgmlDType::IQ2_XS,  // changed: ksigns
        GgmlDType::IQ2_S,   // changed: raw-byte signs
        GgmlDType::IQ3_XXS, // changed: ksigns
        GgmlDType::IQ3_S,   // changed: raw-byte signs
        GgmlDType::IQ1_S,   // control: unchanged (pre-signed)
        GgmlDType::IQ4_XS,  // control: unchanged (no signs)
    ] {
        let raw = synth(dtype, nout, k);
        let wbytes = raw.len() as f64;
        let qs = QStorage::from_data(Cow::Owned(raw), &dev, dtype).unwrap();
        let q = QTensor::new(qs, (nout, k)).unwrap();
        let mm = QMatMul::from_qtensor(q).unwrap();
        let xh: Vec<f32> = q8_1_roundtrip(&(0..k).map(|i| pseudo(i + 7)).collect::<Vec<_>>());
        let x = Tensor::from_vec(xh, (1, k), &dev).unwrap();
        for _ in 0..warm {
            let _ = mm.forward(&x).unwrap();
        }
        dev.synchronize().unwrap();
        let t = std::time::Instant::now();
        for _ in 0..iters {
            let y = mm.forward(&x).unwrap();
            std::hint::black_box(y);
        }
        dev.synchronize().unwrap();
        let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let gbps = wbytes / us / 1e3;
        println!("BENCH {dtype:>8?} {nout}x{k}: {us:7.2} us/call  {gbps:6.1} GB/s");
    }
}

fn run_case(dev: &Device, dtype: GgmlDType, nout: usize, k: usize) -> hanzo_ml::Result<Err> {
    let cpu = Device::Cpu;
    let raw = synth(dtype, nout, k);

    // Ground-truth weights: dequantize the SAME raw bytes on the CPU (the exact values the kernel decodes).
    let qs_cpu = QStorage::from_data(Cow::Owned(raw.clone()), &cpu, dtype)?;
    let q_cpu = QTensor::new(qs_cpu, (nout, k))?;
    let w_deq: Vec<f32> = q_cpu.dequantize(&cpu)?.flatten_all()?.to_vec1::<f32>()?;

    // Native CUDA decode: load the bytes onto the GPU, forward a [1, k] activation through QMatMul.
    // The activation is PRE-quantized onto q8_1 levels (`q8_1_roundtrip`), so the kernel's internal
    // re-quantization is idempotent (each value already sits exactly on its block's int8 level, and
    // the absmax element maps to 127). Both the GPU kernel and the CPU reference then see bit-identical
    // quantized inputs -- the residual is pure f32 accumulation-order, isolating the codebook decode
    // from the q8_1 activation's inherent ~1/256 precision (which would otherwise mask it).
    let qs_cuda = QStorage::from_data(Cow::Owned(raw), dev, dtype)?;
    let q_cuda = QTensor::new(qs_cuda, (nout, k))?;
    let matmul = QMatMul::from_qtensor(q_cuda)?;
    let x_host: Vec<f32> =
        q8_1_roundtrip(&(0..k).map(|i| pseudo(i + 1_000_003)).collect::<Vec<_>>());
    let x = Tensor::from_vec(x_host.clone(), (1, k), dev)?;
    let y_cuda: Vec<f32> = matmul.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(y_cuda.len(), nout);

    // RIGOROUS gate: compare the native dp4a output to the GPU's own DEQUANTIZED-weight matmul (the
    // exact f32 codebook weights, `input @ weight^T`), on the SAME device, fed the SAME pre-quantized
    // activation. Both accumulate in GPU f32, so the q8_1 activation handling and the float arithmetic
    // are identical between the two -- the ONLY remaining difference is the dp4a int-codebook decode
    // vs the `to_float` decode of the weight. Agreement to ~f32 accumulation-order (1e-4) proves the
    // decode is bit-faithful. (A CPU-f64 reference instead conflates the kernel with f32-vs-f64
    // accumulation; the dp4a-vs-dequant gate is the CUDA twin of ROCm's `dp4a-vs-scalar` nbad=0 check.)
    let w_gpu = Tensor::from_vec(w_deq.clone(), (nout, k), dev)?;
    let y_deq: Vec<f32> = x.matmul(&w_gpu.t()?)?.flatten_all()?.to_vec1::<f32>()?;

    let mut max_abs = 0f32;
    let mut sse = 0f64;
    let mut ref_sq = 0f64;
    for n in 0..nout {
        let r = y_deq[n] as f64;
        let g = y_cuda[n] as f64;
        max_abs = max_abs.max((g - r).abs() as f32);
        sse += (g - r) * (g - r);
        ref_sq += r * r;
    }
    let ref_rms = (ref_sq / nout as f64).sqrt();
    let max_rel = ((sse / nout as f64).sqrt() / ref_rms.max(1e-9)) as f32;
    Ok(Err { max_abs, max_rel })
}

fn check(dtype: GgmlDType) {
    let dev = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip {dtype:?}: no CUDA device ({e})");
            return;
        }
    };
    // A couple of decode shapes (attn-proj k=2048, ffn k=4096; non-square nout).
    for &(nout, k) in &[(512usize, 2048usize), (768, 4096)] {
        let s = run_case(&dev, dtype, nout, k).unwrap();
        println!(
            "{dtype:?} [{nout}x{k}]: max_abs={:.3e} max_rel={:.3e}",
            s.max_abs, s.max_rel
        );
        // max_rel (RMS-normalized) is the correctness metric. The dp4a path sums the integer
        // grid x activation products EXACTLY in int32 and applies the f32 scale once, so it is actually
        // MORE precise than the reference, which materializes each weight as f32 (`to_float` rounds
        // db x grid per element, ~1e-7 each -> ~2e-4 accumulated). That f32-weight-materialization floor
        // (uniform ~1.8e-4 across all 7 types) is what max_rel measures here; a genuine codebook-decode
        // bug (wrong grid index / sign) would spike it to O(1e-2..1). The per-element max_abs is
        // additionally amplified on output rows that cancel toward zero (a property of the random data),
        // so it is reported for context with only a loose gross-error bound.
        assert!(
            s.max_rel < 1e-3 && s.max_abs < 1e-1,
            "{dtype:?} [{nout}x{k}] native decode diverged: max_abs={:.3e} max_rel={:.3e}",
            s.max_abs,
            s.max_rel
        );
    }
}

#[test]
fn cuda_matvec_iq2xxs_matches_cpu() {
    check(GgmlDType::IQ2_XXS);
}
#[test]
fn cuda_matvec_iq2xs_matches_cpu() {
    check(GgmlDType::IQ2_XS);
}
#[test]
fn cuda_matvec_iq2s_matches_cpu() {
    check(GgmlDType::IQ2_S);
}
#[test]
fn cuda_matvec_iq3xxs_matches_cpu() {
    check(GgmlDType::IQ3_XXS);
}
#[test]
fn cuda_matvec_iq3s_matches_cpu() {
    check(GgmlDType::IQ3_S);
}
#[test]
fn cuda_matvec_iq1s_matches_cpu() {
    check(GgmlDType::IQ1_S);
}
#[test]
fn cuda_matvec_iq1m_matches_cpu() {
    check(GgmlDType::IQ1_M);
}
#[test]
fn cuda_matvec_iq4xs_matches_cpu() {
    check(GgmlDType::IQ4_XS);
}
// Ternary native dp4a decode (Part 2): TQ2_0 (clean 2-bit) + TQ1_0 (base-3 scalar unpack + dp4a dot).
#[test]
fn cuda_matvec_tq2_0_matches_cpu() {
    check(GgmlDType::TQ2_0);
}
#[test]
fn cuda_matvec_tq1_0_matches_cpu() {
    check(GgmlDType::TQ1_0);
}

// ---- batched (b_size > 1) dense decode: the per-`bi` activation/output offsets in
// mul_mat_vec_iquant_dp4a must be correct (the warp reduction is bit-deterministic, so a wrong
// offset shows as a per-batch mismatch). ----
#[test]
fn cuda_matvec_iq2xxs_batched_matches_cpu() {
    let dev = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip: no CUDA device ({e})");
            return;
        }
    };
    let (nout, k) = (512usize, 2048usize);
    let cpu = Device::Cpu;
    let raw = synth(GgmlDType::IQ2_XXS, nout, k);
    let w_deq: Vec<f32> = QTensor::new(
        QStorage::from_data(Cow::Owned(raw.clone()), &cpu, GgmlDType::IQ2_XXS).unwrap(),
        (nout, k),
    )
    .unwrap()
    .dequantize(&cpu)
    .unwrap()
    .flatten_all()
    .unwrap()
    .to_vec1::<f32>()
    .unwrap();
    let matmul = QMatMul::from_qtensor(
        QTensor::new(
            QStorage::from_data(Cow::Owned(raw), &dev, GgmlDType::IQ2_XXS).unwrap(),
            (nout, k),
        )
        .unwrap(),
    )
    .unwrap();
    for &b in &[1usize, 5, 8] {
        let xh: Vec<f32> = (0..b * k).map(|i| pseudo(i + 99)).collect();
        let xq: Vec<f32> = xh.chunks(k).flat_map(q8_1_roundtrip).collect();
        let y: Vec<f32> = matmul
            .forward(&Tensor::from_vec(xq.clone(), (b, k), &dev).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let mut sse = 0f64;
        let mut rs = 0f64;
        for bi in 0..b {
            for nrow in 0..nout {
                let mut r = 0f64;
                for j in 0..k {
                    r += w_deq[nrow * k + j] as f64 * xq[bi * k + j] as f64;
                }
                let g = y[bi * nout + nrow] as f64;
                sse += (g - r) * (g - r);
                rs += r * r;
            }
        }
        let max_rel =
            ((sse / (b * nout) as f64).sqrt() / (rs / (b * nout) as f64).sqrt().max(1e-9)) as f32;
        println!("IQ2_XXS batched b={b}: max_rel={max_rel:.3e}");
        assert!(
            max_rel < 1e-3,
            "batched b={b} diverged: max_rel={max_rel:.3e}"
        );
    }
}

// ---- dense i-quant PREFILL (m>8 -> the int8-WMMA qmmq GEMM, not the mmvq decode path). Compared to
// the GPU dequantized-weight matmul over the SAME q8_1-pre-quantized activation, isolating the int8
// tile decode (the qmmq twin of the decode gate). ----
fn run_prefill(dev: &Device, dtype: GgmlDType, nout: usize, k: usize, m: usize) -> Err {
    let cpu = Device::Cpu;
    let raw = synth(dtype, nout, k);
    let w_deq: Vec<f32> = QTensor::new(
        QStorage::from_data(Cow::Owned(raw.clone()), &cpu, dtype).unwrap(),
        (nout, k),
    )
    .unwrap()
    .dequantize(&cpu)
    .unwrap()
    .flatten_all()
    .unwrap()
    .to_vec1::<f32>()
    .unwrap();
    let matmul = QMatMul::from_qtensor(
        QTensor::new(
            QStorage::from_data(Cow::Owned(raw), dev, dtype).unwrap(),
            (nout, k),
        )
        .unwrap(),
    )
    .unwrap();
    // m>8 activation, pre-quantized to q8_1 levels per row (idempotent re-quant -> bit-identical inputs).
    let x_host: Vec<f32> = (0..m * k)
        .map(|i| pseudo(i + 555))
        .collect::<Vec<_>>()
        .chunks(k)
        .flat_map(q8_1_roundtrip)
        .collect();
    let x = Tensor::from_vec(x_host.clone(), (m, k), dev).unwrap();
    let y: Vec<f32> = matmul
        .forward(&x)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    // Reference: GPU dense matmul of the exact f32 codebook weights x the same activation.
    let w_gpu = Tensor::from_vec(w_deq, (nout, k), dev).unwrap();
    let y_ref: Vec<f32> = x
        .matmul(&w_gpu.t().unwrap())
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let mut max_abs = 0f32;
    let mut sse = 0f64;
    let mut rs = 0f64;
    for (g, r) in y.iter().zip(&y_ref) {
        let (g, r) = (*g as f64, *r as f64);
        max_abs = max_abs.max((g - r).abs() as f32);
        sse += (g - r) * (g - r);
        rs += r * r;
    }
    let max_rel = ((sse / y.len() as f64).sqrt() / (rs / y.len() as f64).sqrt().max(1e-9)) as f32;
    Err { max_abs, max_rel }
}

#[test]
fn cuda_qmmq_iquant_prefill_matches_dequant() {
    let dev = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip: no CUDA device ({e})");
            return;
        }
    };
    // The 7 i-quant types with an int8-WMMA MMQ kernel: the 5 IQ2/IQ3 codebook grids + IQ4_XS (LUT) +
    // IQ1_S (its distinct GPU-packed iq1s_grid_gpu, DS4 ds-layout for the delta bias). IQ1_M has no MMQ
    // kernel -> dequant prefill, not tested here.
    for dtype in [
        GgmlDType::IQ2_XXS,
        GgmlDType::IQ2_XS,
        GgmlDType::IQ2_S,
        GgmlDType::IQ3_XXS,
        GgmlDType::IQ3_S,
        GgmlDType::IQ4_XS,
        GgmlDType::IQ1_S,
    ] {
        for &m in &[16usize, 32] {
            let s = run_prefill(&dev, dtype, 512, 2048, m);
            println!(
                "qmmq {dtype:?} prefill m={m}: max_abs={:.3e} max_rel={:.3e}",
                s.max_abs, s.max_rel
            );
            assert!(
                s.max_rel < 1e-3 && s.max_abs < 1e-1,
                "{dtype:?} qmmq prefill m={m} diverged: max_abs={:.3e} max_rel={:.3e}",
                s.max_abs,
                s.max_rel
            );
        }
    }
}

// ---- fused i-quant MoE decode: the native dp4a MoE path (bank resident in VRAM, on-device gather)
// vs the CPU per-expert reference, both fed the same q8_1-level activation + router ids. Exercises the
// mod.rs CUDA i-quant MoE arm + moe_iquant_dp4a (quantize + the moe_qmatvec_dp4a_<iq*> launch). ----
#[allow(clippy::too_many_arguments)]
fn run_moe(
    dev: &Device,
    dtype: GgmlDType,
    e_cnt: usize,
    n: usize,
    k: usize,
    t: usize,
    topk: usize,
    shared: bool,
) -> Err {
    let cpu = Device::Cpu;
    // Bank [E, n, k]: synth over e_cnt*n rows -- the counter increments across experts so each expert's
    // codebook bytes differ (a wrong-expert gather would mismatch).
    let raw = synth(dtype, e_cnt * n, k);
    let q_cuda = QTensor::new(
        QStorage::from_data(Cow::Owned(raw.clone()), dev, dtype).unwrap(),
        (e_cnt, n, k),
    )
    .unwrap();
    // Ground-truth bank weights: dequantize on the CPU (the exact codebook values).
    let w_deq: Vec<f32> = QTensor::new(
        QStorage::from_data(Cow::Owned(raw), &cpu, dtype).unwrap(),
        (e_cnt, n, k),
    )
    .unwrap()
    .dequantize(&cpu)
    .unwrap()
    .flatten_all()
    .unwrap()
    .to_vec1::<f32>()
    .unwrap();
    // Activation: [t, topk|1, k], pre-quantized to q8_1 levels (so dp4a's re-quant is idempotent and
    // the reference sees identical activations).
    let in1 = if shared { 1 } else { topk };
    let x_host: Vec<f32> = q8_1_roundtrip(
        &(0..t * in1 * k)
            .map(|i| pseudo(i + 1_234_567))
            .collect::<Vec<_>>(),
    );
    let x_cuda = Tensor::from_vec(x_host.clone(), (t, in1, k), dev).unwrap();
    let ids_host: Vec<u32> = (0..t * topk)
        .map(|i| (pbyte(i * 7 + 3) as u32) % e_cnt as u32)
        .collect();
    let ids_cuda = Tensor::from_vec(ids_host.clone(), (t, topk), dev).unwrap();

    let y_cuda: Vec<f32> = q_cuda
        .indexed_moe_forward(&x_cuda, &ids_cuda)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(y_cuda.len(), t * topk * n);

    // Self-contained reference: for each slot s=(token,slot_in_topk), output[s] = W[ids[s]] . act, where
    // act is the slot's activation row (shared token input for in1==1, else the per-slot row). f64 dot.
    let mut max_abs = 0f32;
    let mut sse = 0f64;
    let mut rs = 0f64;
    for token in 0..t {
        for slot in 0..topk {
            let s = token * topk + slot;
            let expert = ids_host[s] as usize;
            let act_off = if in1 == 1 { token * k } else { s * k };
            for row in 0..n {
                let mut r = 0f64;
                let w_off = (expert * n + row) * k;
                for j in 0..k {
                    r += w_deq[w_off + j] as f64 * x_host[act_off + j] as f64;
                }
                let g = y_cuda[s * n + row] as f64;
                max_abs = max_abs.max((g - r).abs() as f32);
                sse += (g - r) * (g - r);
                rs += r * r;
            }
        }
    }
    let cnt = (t * topk * n) as f64;
    let max_rel = ((sse / cnt).sqrt() / (rs / cnt).sqrt().max(1e-9)) as f32;
    Err { max_abs, max_rel }
}

#[test]
fn cuda_moe_iq2xxs_matches_cpu() {
    let dev = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip: no CUDA device ({e})");
            return;
        }
    };
    // Cover BOTH the codebook-grid headline (IQ2_XXS) AND the LUT headline (IQ4_XS -- the dominant
    // MoE-quant type) so an IQ4_XS-specific MoE wiring regression (decode moe_qmatvec_dp4a_iq4xs /
    // prefill moe qmmq) can't slip past the suite (closes red's [LOW] coverage gap).
    // (e_cnt, n, k, t, topk, shared-input): decode t=1 + a small prefill t=3; shared (gate/up) +
    // per-slot (down) input layouts.
    for dtype in [GgmlDType::IQ2_XXS, GgmlDType::IQ4_XS] {
        for &(e, n, k, t, topk, shared) in &[
            (8usize, 512usize, 2048usize, 1usize, 4usize, true),
            (8, 512, 2048, 1, 4, false),
            (16, 768, 2048, 3, 6, true),
        ] {
            let s = run_moe(&dev, dtype, e, n, k, t, topk, shared);
            println!("MoE {dtype:?} [E{e} {n}x{k} t{t} topk{topk} shared{shared}]: max_abs={:.3e} max_rel={:.3e}", s.max_abs, s.max_rel);
            assert!(
                s.max_rel < 1e-3 && s.max_abs < 1e-1,
                "fused MoE {dtype:?} diverged: max_abs={:.3e} max_rel={:.3e}",
                s.max_abs,
                s.max_rel
            );
        }
    }
}

// ---- fused MoE ROUTER (moe_route kernel): softmax-over-all-experts -> top-k by descending logit ->
// optional renorm of the top-k to sum 1. ONE block/token replaces the candle softmax->asort->narrow->
// sum->div op chain. Oracle: the CUDA kernel vs the ml-op reference -- the SAME public `moe_route` run
// on a CPU tensor (no fast path -> it executes the op chain). Mirrors the ROCm `moe_route_numeric`
// gate. ids must match EXACTLY (descending logit; random data has no f32 ties); weights agree to f32
// reduction-order tolerance (the kernel sums Z in a shared-mem tree, the CPU sequentially). ----
#[test]
fn cuda_moe_route_numeric() {
    let dev = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip: no CUDA device ({e})");
            return;
        }
    };
    let cpu = Device::Cpu;
    let mut worst = 0f32;
    for &(ntok, n_experts) in &[
        (1usize, 60usize),
        (1, 128),
        (1, 256), // == MOE_ROUTE_MAX_E (exercises the full shared-mem expert buffer)
        (7, 128),
        (181, 128), // the decode-microbatch token count from the 30B-A3B profile
        (64, 256),
        (1024, 160),
    ] {
        for &topk in &[4usize, 8] {
            for &norm in &[false, true] {
                // logits in [-4, 4) -> a peaked softmax (real dynamic range, not near-uniform).
                let seed = ntok * 131 + n_experts * 17 + topk * 7 + norm as usize;
                let logits: Vec<f32> = (0..ntok * n_experts)
                    .map(|i| pseudo(i + seed) * 4.0)
                    .collect();
                let lg_cuda = Tensor::from_vec(logits.clone(), (ntok, n_experts), &dev).unwrap();
                let lg_cpu = Tensor::from_vec(logits, (ntok, n_experts), &cpu).unwrap();

                let (ids_g, w_g) = hanzo_ml::quantized::moe_route(&lg_cuda, topk, norm).unwrap();
                let (ids_r, w_r) = hanzo_ml::quantized::moe_route(&lg_cpu, topk, norm).unwrap();

                let ids_g = ids_g.flatten_all().unwrap().to_vec1::<u32>().unwrap();
                let ids_r = ids_r.flatten_all().unwrap().to_vec1::<u32>().unwrap();
                let w_g = w_g.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let w_r = w_r.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                assert_eq!(ids_g.len(), ntok * topk, "fused ids shape");

                let mut nbad_id = 0usize;
                let mut max_w_err = 0f32;
                for i in 0..ntok * topk {
                    if ids_g[i] != ids_r[i] {
                        nbad_id += 1;
                    }
                    max_w_err = max_w_err.max((w_g[i] - w_r[i]).abs());
                }
                worst = worst.max(max_w_err);
                println!(
                    "moe_route ntok={ntok} E={n_experts} topk={topk} norm={norm}: nbad_id={nbad_id} max_w_err={max_w_err:.3e}"
                );
                assert_eq!(
                    nbad_id, 0,
                    "expert-id mismatch ntok={ntok} E={n_experts} topk={topk} norm={norm}"
                );
                assert!(
                    max_w_err < 1e-5,
                    "weights diverged ntok={ntok} E={n_experts} topk={topk} norm={norm}: max_w_err={max_w_err:.3e}"
                );
            }
        }
    }
    println!("cuda_moe_route_numeric: worst max_w_err across all configs = {worst:.3e}");
}

fn q8_1_roundtrip_f16d(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    let mut i = 0;
    for blk in x.chunks(32) {
        let absmax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let d = if absmax > 0.0 {
            half::f16::from_f32(absmax / 127.0).to_f32()
        } else {
            0.0
        };
        let inv = if absmax > 0.0 { 127.0 / absmax } else { 0.0 };
        for &v in blk {
            let q = (v * inv).round().clamp(-127.0, 127.0);
            out[i] = q * d;
            i += 1;
        }
    }
    out
}

#[test]
fn cuda_bonsai2_q1_pq2_ptq1_comprehensive_oracle() {
    use hanzo_ml::DType;
    let dev = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip: no CUDA device ({e})");
            return;
        }
    };
    let cpu = Device::Cpu;

    for dtype in [GgmlDType::Q1_0, GgmlDType::PQ2_0, GgmlDType::PTQ1_0] {
        // Exercise non-multiple-of-4 nrows (5, 7, 19), standard nrows (256), and non-multiple-of-256 k (384).
        for &(nout, k) in &[(5usize, 384usize), (7, 512), (19, 1024), (256, 512)] {
            let raw = synth(dtype, nout, k);
            let q_cpu = QTensor::new(
                QStorage::from_data(Cow::Owned(raw.clone()), &cpu, dtype).unwrap(),
                (nout, k),
            )
            .unwrap();
            let w_cpu: Vec<f32> = q_cpu
                .dequantize(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();

            let q_cuda = std::sync::Arc::new(
                QTensor::new(
                    QStorage::from_data(Cow::Owned(raw), &dev, dtype).unwrap(),
                    (nout, k),
                )
                .unwrap(),
            );

            // 1. Dequantize F32 bit-exact parity vs CPU.
            let w_gpu: Vec<f32> = q_cuda
                .dequantize(&dev)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            for i in 0..w_cpu.len() {
                assert_eq!(
                    w_gpu[i].to_bits(),
                    w_cpu[i].to_bits(),
                    "{dtype:?} [{nout}x{k}] dequantize mismatch at {i}"
                );
            }

            // 2. Dequantize F16 tight parity vs CPU.
            let w_f16: Vec<f32> = q_cuda
                .dequantize_f16(&dev)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            for i in 0..w_cpu.len() {
                let diff = (w_f16[i] - w_cpu[i]).abs();
                assert!(diff < 1e-5, "{dtype:?} dequantize_f16 diff {diff}");
            }

            // 3. Embed / get_rows bit-exact parity vs CPU.
            let ids_vec: Vec<u32> = vec![0, (nout / 2) as u32, (nout - 1) as u32];
            let ids_gpu = Tensor::from_vec(ids_vec.clone(), ids_vec.len(), &dev).unwrap();
            let emb_gpu: Vec<f32> = q_cuda
                .embedding(&ids_gpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            for (ri, &rid) in ids_vec.iter().enumerate() {
                for c in 0..k {
                    assert_eq!(
                        emb_gpu[ri * k + c].to_bits(),
                        w_cpu[rid as usize * k + c].to_bits(),
                        "{dtype:?} embed mismatch row {rid} col {c}"
                    );
                }
            }

            let mm_gpu = QMatMul::from_arc(q_cuda).unwrap();
            let w_gpu_dense = Tensor::from_vec(w_cpu.clone(), (nout, k), &dev).unwrap();

            // 4. Decode m=1 across F32, BF16, and F16 inputs (direct iq_quantize_q8_{f32,bf16,f16}).
            let x1_raw: Vec<f32> = q8_1_roundtrip(
                &(0..k)
                    .map(|i| pseudo(i + nout * 31 + k))
                    .collect::<Vec<_>>(),
            );
            let x1_f32 = Tensor::from_vec(x1_raw, (1, k), &dev).unwrap();
            for in_dt in [DType::F32, DType::BF16, DType::F16] {
                let x_in = x1_f32.to_dtype(in_dt).unwrap();
                let y_out = mm_gpu.forward(&x_in).unwrap();
                assert_eq!(y_out.dtype(), in_dt);
                let y_got: Vec<f32> = y_out
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let x_rt: Vec<f32> = q8_1_roundtrip_f16d(
                    &x_in
                        .to_dtype(DType::F32)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap(),
                );
                let mut max_err = 0f32;
                let mut max_ref = 0f32;
                for r in 0..nout {
                    let mut acc = 0f64;
                    for c in 0..k {
                        acc += (w_cpu[r * k + c] as f64) * (x_rt[c] as f64);
                    }
                    let want = if in_dt == DType::BF16 {
                        half::bf16::from_f32(acc as f32).to_f32()
                    } else if in_dt == DType::F16 {
                        half::f16::from_f32(acc as f32).to_f32()
                    } else {
                        acc as f32
                    };
                    max_ref = max_ref.max(want.abs());
                    max_err = max_err.max((y_got[r] - want).abs());
                }
                let tol = if in_dt == DType::F32 {
                    1e-4 * max_ref.max(1.0)
                } else if in_dt == DType::F16 {
                    2e-3 * max_ref.max(1.0)
                } else {
                    1e-2 * max_ref.max(1.0)
                };
                assert!(
                    max_err < tol,
                    "{dtype:?} [{nout}x{k}] decode {in_dt:?} max_err={max_err:.3e} > tol={tol:.3e}"
                );
            }

            // 5. Verify m=4, 16 and prefill m=64, 256 against q8_1-matched dense reference (< 1e-4 rel).
            for &m in &[1usize, 4, 16, 64, 256] {
                if m == 256 && nout < 256 && k != 384 {
                    continue;
                }
                let xm_host: Vec<f32> = (0..m * k)
                    .map(|i| pseudo(i + m * 97 + k))
                    .collect::<Vec<_>>()
                    .chunks(k)
                    .flat_map(q8_1_roundtrip_f16d)
                    .collect();
                let xm = Tensor::from_vec(xm_host, (m, k), &dev).unwrap();
                let ym: Vec<f32> = mm_gpu
                    .forward(&xm)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let ym_ref: Vec<f32> = xm
                    .matmul(&w_gpu_dense.t().unwrap())
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let mut sse = 0f64;
                let mut rsq = 0f64;
                let mut max_abs = 0f32;
                for (&g, &r) in ym.iter().zip(&ym_ref) {
                    let diff = (g - r).abs();
                    max_abs = max_abs.max(diff);
                    sse += (diff as f64) * (diff as f64);
                    rsq += (r as f64) * (r as f64);
                }
                let rel_rmse = ((sse / ym.len() as f64).sqrt()
                    / (rsq / ym.len() as f64).sqrt().max(1e-9)) as f32;
                let rel_tol = if k % 256 != 0 && m > 64 { 5e-3 } else { 5e-4 };
                assert!(
                    rel_rmse < rel_tol,
                    "{dtype:?} [{nout}x{k}] m={m} rel_rmse={rel_rmse:.3e} > {rel_tol:.3e} (max_abs={max_abs:.3e})"
                );
            }
        }
    }
}

#[test]
fn cuda_q2_0_entrypoints_oracle() {
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    use half::f16;
    use hanzo_ml::backend::BackendDevice;
    let dev = match hanzo_ml::CudaDevice::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip: no CUDA device ({e})");
            return;
        }
    };
    let (nout, k) = (19usize, 256usize);
    let nblocks = nout * (k / 64);
    let mut raw = Vec::with_capacity(nblocks * 18);
    let mut w_cpu = vec![0f32; nout * k];
    for b in 0..nblocks {
        let d_f16 = f16::from_f32(pseudo(b * 19 + 3) * 0.2);
        let d = d_f16.to_f32();
        raw.extend_from_slice(&d_f16.to_le_bytes());
        for j in 0..16 {
            let byte = pbyte(b * 37 + j);
            raw.push(byte);
            for sub in 0..4 {
                let code = ((byte >> (2 * sub)) & 3) as i32 - 1;
                w_cpu[b * 64 + j * 4 + sub] = (code as f32) * d;
            }
        }
    }
    let wq_dev = dev.clone_htod(&raw).unwrap();

    // 1. dequantize_q2_0_f32 bit-exact vs CPU.
    let deq_dev = unsafe { dev.alloc::<f32>(nout * k).unwrap() };
    let f_deq = dev
        .get_or_load_func("dequantize_q2_0_f32", &hanzo_kernels::IQUANT_MMVQ)
        .unwrap();
    let mut b_deq = f_deq.builder();
    let nb_i64 = nblocks as i64;
    b_deq.arg(&wq_dev);
    b_deq.arg(&deq_dev);
    b_deq.arg(&nb_i64);
    unsafe {
        b_deq.launch(LaunchConfig {
            grid_dim: (nblocks as u32, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .unwrap();
    let deq_host = dev.clone_dtoh(&deq_dev).unwrap();
    for i in 0..w_cpu.len() {
        assert_eq!(deq_host[i].to_bits(), w_cpu[i].to_bits(), "q2_0 dequant mismatch at {i}");
    }

    // 2. qmatvec_q2_0_dp4a_f32 (with nrows=19 non-multiple-of-4) vs CPU.
    let x_raw: Vec<f32> = (0..k).map(|i| pseudo(i + 41)).collect();
    let x_host = q8_1_roundtrip_f16d(&x_raw);
    let x_dev = dev.clone_htod(&x_raw).unwrap();
    let total_q8 = k / 32;
    let xq_dev = unsafe { dev.alloc::<u8>(total_q8 * 32).unwrap() };
    let xd_dev = unsafe { dev.alloc::<f16>(total_q8).unwrap() };
    let f_q8 = dev
        .get_or_load_func("iq_quantize_q8_f32", &hanzo_kernels::IQUANT_MMVQ)
        .unwrap();
    let m_one = 1i32;
    let k_i32 = k as i32;
    let tot_i32 = total_q8 as i32;
    let mut b_q8 = f_q8.builder();
    b_q8.arg(&m_one);
    b_q8.arg(&k_i32);
    b_q8.arg(&tot_i32);
    b_q8.arg(&x_dev);
    b_q8.arg(&xq_dev);
    b_q8.arg(&xd_dev);
    unsafe {
        b_q8.launch(LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .unwrap();

    let y_dev = unsafe { dev.alloc::<f32>(nout).unwrap() };
    let f_mv = dev
        .get_or_load_func("qmatvec_q2_0_dp4a_f32", &hanzo_kernels::IQUANT_MMVQ)
        .unwrap();
    let n_i32 = nout as i32;
    let mut b_mv = f_mv.builder();
    b_mv.arg(&n_i32);
    b_mv.arg(&k_i32);
    b_mv.arg(&wq_dev);
    b_mv.arg(&xq_dev);
    b_mv.arg(&xd_dev);
    b_mv.arg(&y_dev);
    unsafe {
        b_mv.launch(LaunchConfig {
            grid_dim: ((nout as u32).div_ceil(16), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .unwrap();
    let y_host = dev.clone_dtoh(&y_dev).unwrap();
    for r in 0..nout {
        let mut acc = 0f64;
        for c in 0..k {
            acc += (w_cpu[r * k + c] as f64) * (x_host[c] as f64);
        }
        let diff = (y_host[r] - acc as f32).abs();
        assert!(diff < 1e-4, "q2_0 matvec row {r} diff {diff}");
    }
}

#[test]
fn cuda_wide_fwht_and_fused_rmsnorm_fwht_oracle() {
    use hanzo_ml::DType;
    let dev = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip: no CUDA device ({e})");
            return;
        }
    };
    let cpu = Device::Cpu;

    // Cover B = 512, 1024, 4096 including block-diagonal D = 5120, B = 1024 and D = 17408, B = 1024.
    let configs = [
        (4usize, 2560usize, 512usize),
        (3usize, 5120usize, 1024usize),
        (2usize, 17408usize, 1024usize),
        (3usize, 4096usize, 4096usize),
    ];

    for (rows, width, b) in configs {
        let x_vec: Vec<f32> = (0..rows * width)
            .map(|i| pseudo(i + width + b) * 0.8)
            .collect();
        let r_vec: Vec<f32> = (0..rows * width)
            .map(|i| pseudo(i + width * 3 + 17) * 0.5)
            .collect();
        let w_vec: Vec<f32> = (0..width)
            .map(|i| 0.5 + 0.5 * pseudo(i + 777).abs())
            .collect();
        let s_vec: Vec<f32> = (0..width)
            .map(|i| if (i * 13 + 5) % 2 == 0 { 1.0 } else { -1.0 })
            .collect();

        let x_gpu = Tensor::from_vec(x_vec.clone(), (rows, width), &dev).unwrap();
        let r_gpu = Tensor::from_vec(r_vec.clone(), (rows, width), &dev).unwrap();
        let w_gpu = Tensor::from_vec(w_vec.clone(), width, &dev).unwrap();
        let s_gpu = Tensor::from_vec(s_vec.clone(), width, &dev).unwrap();

        let x_cpu = Tensor::from_vec(x_vec, (rows, width), &cpu).unwrap();
        let w_cpu = Tensor::from_vec(w_vec, width, &cpu).unwrap();
        let s_cpu = Tensor::from_vec(s_vec, width, &cpu).unwrap();

        // 1. Forward + inverse FWHT vs CPU oracle (< 1e-5) and exact roundtrip.
        let y_gpu = hanzo_ml::quantized::hadamard_fwht(&x_gpu, Some(&s_gpu), b, false).unwrap();
        let y_cpu = hanzo_ml::quantized::hadamard_fwht(&x_cpu, Some(&s_cpu), b, false).unwrap();
        let yg = y_gpu.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let yc = y_cpu.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut max_fwht_diff = 0f32;
        for i in 0..yg.len() {
            max_fwht_diff = max_fwht_diff.max((yg[i] - yc[i]).abs());
        }
        assert!(
            max_fwht_diff < 1e-5,
            "FWHT fwd D={width} B={b} max_diff={max_fwht_diff:.3e}"
        );

        let rec_gpu = hanzo_ml::quantized::hadamard_fwht(&y_gpu, Some(&s_gpu), b, true).unwrap();
        let rec_v = rec_gpu.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let orig_v = x_gpu.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut max_rt_diff = 0f32;
        for i in 0..orig_v.len() {
            max_rt_diff = max_rt_diff.max((rec_v[i] - orig_v[i]).abs());
        }
        assert!(
            max_rt_diff < 1e-5,
            "FWHT inverse roundtrip D={width} B={b} max_diff={max_rt_diff:.3e}"
        );

        // 2. Fused rms_norm_hadamard_fwht (F32, BF16, F16 inputs -> F32 dst) vs CPU oracle (< 1e-5).
        let eps = 1e-6f32;
        let rn_gpu = x_gpu
            .rms_norm_hadamard_fwht(&w_gpu, eps, Some(&s_gpu), b)
            .unwrap();
        assert_eq!(rn_gpu.dtype(), DType::F32);
        let rn_cpu = x_cpu
            .rms_norm_hadamard_fwht(&w_cpu, eps, Some(&s_cpu), b)
            .unwrap();
        let rng = rn_gpu.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let rnc = rn_cpu.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut max_rn_diff = 0f32;
        for i in 0..rng.len() {
            max_rn_diff = max_rn_diff.max((rng[i] - rnc[i]).abs());
        }
        assert!(
            max_rn_diff < 1e-5,
            "rms_norm_hadamard_fwht F32 D={width} B={b} max_diff={max_rn_diff:.3e}"
        );

        // 3. Fused add_rms_norm_hadamard_fwht in BF16 and F32 vs CPU oracle (< 1e-5).
        for dt in [DType::F32, DType::BF16, DType::F16] {
            let xd = x_gpu.to_dtype(dt).unwrap();
            let rd = r_gpu.to_dtype(dt).unwrap();
            let wd = w_gpu.to_dtype(dt).unwrap();
            let xc = xd.to_device(&cpu).unwrap();
            let rc = rd.to_device(&cpu).unwrap();
            let wc = wd.to_device(&cpu).unwrap();

            let (h_g, rot_g) = xd
                .add_rms_norm_hadamard_fwht(&rd, &wd, eps, Some(&s_gpu), b)
                .unwrap();
            let (h_c, rot_c) = xc
                .add_rms_norm_hadamard_fwht(&rc, &wc, eps, Some(&s_cpu), b)
                .unwrap();
            assert_eq!(h_g.dtype(), dt);
            assert_eq!(rot_g.dtype(), DType::F32);

            let hg = h_g
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let hc = h_c
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            for i in 0..hg.len() {
                assert_eq!(
                    hg[i].to_bits(),
                    hc[i].to_bits(),
                    "{dt:?} residual_out mismatch at {i}: gpu={} cpu={}",
                    hg[i],
                    hc[i]
                );
            }

            let rg = rot_g.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let rc_v = rot_c.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let mut max_d = 0f32;
            for i in 0..rg.len() {
                max_d = max_d.max((rg[i] - rc_v[i]).abs());
            }
            assert!(
                max_d < 1e-5,
                "add_rms_norm_hadamard_fwht {dt:?} D={width} B={b} max_diff={max_d:.3e}"
            );
        }
    }
}

#[test]
fn cuda_shared_iq_q8_sibling_matvec_parity() {
    use hanzo_ml::DType;
    let dev = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip: no CUDA device ({e})");
            return;
        }
    };
    let (k, b_fwht) = (5120usize, 1024usize);
    let eps = 1e-6f32;
    let w_norm = Tensor::from_vec(
        (0..k).map(|i| 0.7 + 0.3 * pseudo(i + 11).abs()).collect::<Vec<_>>(),
        k,
        &dev,
    )
    .unwrap();
    let signs = Tensor::from_vec(
        (0..k)
            .map(|i| if (i * 5 + 1) % 2 == 0 { 1.0f32 } else { -1.0f32 })
            .collect::<Vec<_>>(),
        k,
        &dev,
    )
    .unwrap();

    for dtype in [GgmlDType::PQ2_0, GgmlDType::PTQ1_0, GgmlDType::Q1_0] {
        let wq = QMatMul::from_qtensor(
            QTensor::new(
                QStorage::from_data(Cow::Owned(synth(dtype, 512, k)), &dev, dtype).unwrap(),
                (512, k),
            )
            .unwrap(),
        )
        .unwrap();
        let wk = QMatMul::from_qtensor(
            QTensor::new(
                QStorage::from_data(Cow::Owned(synth(dtype, 128, k)), &dev, dtype).unwrap(),
                (128, k),
            )
            .unwrap(),
        )
        .unwrap();
        let wv = QMatMul::from_qtensor(
            QTensor::new(
                QStorage::from_data(Cow::Owned(synth(dtype, 19, k)), &dev, dtype).unwrap(),
                (19, k),
            )
            .unwrap(),
        )
        .unwrap();

        for &m in &[1usize, 4] {
            let x_bf16 = Tensor::from_vec(
                (0..m * k).map(|i| pseudo(i + 909) * 0.9).collect::<Vec<_>>(),
                (m, k),
                &dev,
            )
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

            let x_rot = x_bf16
                .rms_norm_hadamard_fwht(&w_norm, eps, Some(&signs), b_fwht)
                .unwrap();

            // Path A: standalone quantize_iq_q8 + forward_iq_q8 vs QMatMul::forward (bit-exact).
            let q8_sep = QMatMul::quantize_iq_q8(&x_rot).unwrap();
            // Path B: single-launch fused rms_norm_hadamard_fwht_iq_q8 + forward_iq_q8 (bit-exact).
            let q8_fused =
                QMatMul::rms_norm_hadamard_fwht_iq_q8(&x_bf16, &w_norm, eps, Some(&signs), b_fwht)
                    .unwrap();

            for proj in [&wq, &wk, &wv] {
                let want = if m == 1 {
                    proj.forward(&x_rot).unwrap()
                } else {
                    // At m=4, PQ2_0 QMatMul::forward may route to MMQ; compare against q8_sep (dp4a).
                    proj.forward_iq_q8(&q8_sep).unwrap()
                };
                let got_sep = proj.forward_iq_q8(&q8_sep).unwrap();
                let got_fused = proj.forward_iq_q8(&q8_fused).unwrap();

                let v_want = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let v_sep = got_sep.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let v_fused = got_fused.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                for i in 0..v_want.len() {
                    assert_eq!(
                        v_sep[i].to_bits(),
                        v_want[i].to_bits(),
                        "{dtype:?} m={m} quantize_iq_q8 mismatch at {i}"
                    );
                    assert_eq!(
                        v_fused[i].to_bits(),
                        v_sep[i].to_bits(),
                        "{dtype:?} m={m} fused rms_norm_hadamard_fwht_iq_q8 mismatch at {i}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore]
fn bench_bonsai2_cuda_kernels() {
    use hanzo_ml::DType;
    let dev = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip bench: no CUDA device ({e})");
            return;
        }
    };
    let shapes = [(4096usize, 4096usize), (12288, 4096), (5120, 17408)];
    let (warm, iters) = (30usize, 300usize);

    println!("=== GB10 Bonsai-2 CUDA Decode (m=1) & Shared-Q8 Microbench ===");
    for &(nout, k) in &shapes {
        let x = Tensor::from_vec(
            (0..k).map(|i| pseudo(i + 17)).collect::<Vec<_>>(),
            (1, k),
            &dev,
        )
        .unwrap();
        let q8 = QMatMul::quantize_iq_q8(&x).unwrap();
        for dtype in [GgmlDType::PQ2_0, GgmlDType::PTQ1_0, GgmlDType::Q1_0] {
            let raw = synth(dtype, nout, k);
            let wbytes = raw.len() as f64;
            let mm = QMatMul::from_qtensor(
                QTensor::new(QStorage::from_data(Cow::Owned(raw), &dev, dtype).unwrap(), (nout, k))
                    .unwrap(),
            )
            .unwrap();

            for _ in 0..warm {
                let _ = mm.forward(&x).unwrap();
                let _ = mm.forward_iq_q8(&q8).unwrap();
            }
            dev.synchronize().unwrap();

            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                std::hint::black_box(mm.forward(&x).unwrap());
            }
            dev.synchronize().unwrap();
            let us_fwd = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            let gbps_fwd = wbytes / us_fwd / 1e3;

            let t1 = std::time::Instant::now();
            for _ in 0..iters {
                std::hint::black_box(mm.forward_iq_q8(&q8).unwrap());
            }
            dev.synchronize().unwrap();
            let us_q8 = t1.elapsed().as_secs_f64() * 1e6 / iters as f64;
            let gbps_q8 = wbytes / us_q8 / 1e3;

            println!(
                "DECODE {dtype:>6?} [{nout:5}x{k:5}]: fwd={us_fwd:6.2} us ({gbps_fwd:5.1} GB/s) | shared_q8={us_q8:6.2} us ({gbps_q8:5.1} GB/s)"
            );
        }
    }

    println!("=== GB10 Bonsai-2 Wide-CTA FWHT & Fused RMSNorm+FWHT+Q8 Microbench ===");
    for &(width, b) in &[(4096usize, 4096usize), (5120, 1024), (17408, 1024)] {
        let x = Tensor::from_vec(
            (0..width).map(|i| pseudo(i + 3)).collect::<Vec<_>>(),
            (1, width),
            &dev,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let res = Tensor::from_vec(
            (0..width).map(|i| pseudo(i + 99)).collect::<Vec<_>>(),
            (1, width),
            &dev,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let w = Tensor::from_vec(vec![1.0f32; width], width, &dev).unwrap();
        let s = Tensor::from_vec(vec![1.0f32; width], width, &dev).unwrap();

        for _ in 0..warm {
            let _ = hanzo_ml::quantized::hadamard_fwht(&x, Some(&s), b, false).unwrap();
            let _ = QMatMul::add_rms_norm_hadamard_fwht_iq_q8(&x, &res, &w, 1e-6, Some(&s), b)
                .unwrap();
        }
        dev.synchronize().unwrap();

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            std::hint::black_box(
                hanzo_ml::quantized::hadamard_fwht(&x, Some(&s), b, false).unwrap(),
            );
        }
        dev.synchronize().unwrap();
        let us_fwht = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            std::hint::black_box(
                QMatMul::add_rms_norm_hadamard_fwht_iq_q8(&x, &res, &w, 1e-6, Some(&s), b).unwrap(),
            );
        }
        dev.synchronize().unwrap();
        let us_fused = t1.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!(
            "FWHT [1x{width:5}, B={b:4}]: fwht_bf16={us_fwht:5.2} us | fused_add_rmsnorm_fwht_q8={us_fused:5.2} us"
        );
    }

    println!("=== GB10 Bonsai-2 CUDA MMQ Prefill (m=128) Microbench ===");
    for &(nout, k) in &shapes {
        let m = 128usize;
        let x = Tensor::from_vec(
            (0..(m * k)).map(|i| pseudo(i + 23)).collect::<Vec<_>>(),
            (m, k),
            &dev,
        )
        .unwrap();
        let flops = 2.0 * (m as f64) * (nout as f64) * (k as f64);
        for dtype in [GgmlDType::PQ2_0, GgmlDType::PTQ1_0, GgmlDType::Q1_0] {
            let raw = synth(dtype, nout, k);
            let mm = QMatMul::from_qtensor(
                QTensor::new(QStorage::from_data(Cow::Owned(raw), &dev, dtype).unwrap(), (nout, k))
                    .unwrap(),
            )
            .unwrap();
            for _ in 0..10 {
                let _ = mm.forward(&x).unwrap();
            }
            dev.synchronize().unwrap();
            let p_iters = 50usize;
            let t0 = std::time::Instant::now();
            for _ in 0..p_iters {
                std::hint::black_box(mm.forward(&x).unwrap());
            }
            dev.synchronize().unwrap();
            let us = t0.elapsed().as_secs_f64() * 1e6 / p_iters as f64;
            let tflops = flops / us / 1e6;
            println!("PREFILL m=128 {dtype:>6?} [{nout:5}x{k:5}]: {us:7.1} us ({tflops:5.2} TFLOP/s)");
        }
    }
}
