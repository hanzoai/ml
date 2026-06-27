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
    let x_host: Vec<f32> = q8_1_roundtrip(&(0..k).map(|i| pseudo(i + 1_000_003)).collect::<Vec<_>>());
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
        println!("{dtype:?} [{nout}x{k}]: max_abs={:.3e} max_rel={:.3e}", s.max_abs, s.max_rel);
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
        QTensor::new(QStorage::from_data(Cow::Owned(raw), &dev, GgmlDType::IQ2_XXS).unwrap(), (nout, k))
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
        let max_rel = ((sse / (b * nout) as f64).sqrt() / (rs / (b * nout) as f64).sqrt().max(1e-9)) as f32;
        println!("IQ2_XXS batched b={b}: max_rel={max_rel:.3e}");
        assert!(max_rel < 1e-3, "batched b={b} diverged: max_rel={max_rel:.3e}");
    }
}

// ---- fused i-quant MoE decode: the native dp4a MoE path (bank resident in VRAM, on-device gather)
// vs the CPU per-expert reference, both fed the same q8_1-level activation + router ids. Exercises the
// mod.rs CUDA i-quant MoE arm + moe_iquant_dp4a (quantize + the moe_qmatvec_dp4a_<iq*> launch). ----
fn run_moe(dev: &Device, e_cnt: usize, n: usize, k: usize, t: usize, topk: usize, shared: bool) -> Err {
    let cpu = Device::Cpu;
    // Bank [E, n, k]: synth over e_cnt*n rows -- the counter increments across experts so each expert's
    // codebook bytes differ (a wrong-expert gather would mismatch).
    let raw = synth(GgmlDType::IQ2_XXS, e_cnt * n, k);
    let q_cuda = QTensor::new(
        QStorage::from_data(Cow::Owned(raw.clone()), dev, GgmlDType::IQ2_XXS).unwrap(),
        (e_cnt, n, k),
    )
    .unwrap();
    // Ground-truth bank weights: dequantize on the CPU (the exact codebook values).
    let w_deq: Vec<f32> = QTensor::new(
        QStorage::from_data(Cow::Owned(raw), &cpu, GgmlDType::IQ2_XXS).unwrap(),
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
    let x_host: Vec<f32> =
        q8_1_roundtrip(&(0..t * in1 * k).map(|i| pseudo(i + 1_234_567)).collect::<Vec<_>>());
    let x_cuda = Tensor::from_vec(x_host.clone(), (t, in1, k), dev).unwrap();
    let ids_host: Vec<u32> = (0..t * topk).map(|i| (pbyte(i * 7 + 3) as u32) % e_cnt as u32).collect();
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
    // (e_cnt, n, k, t, topk, shared-input). Decode t=1 + a small prefill t=3; shared (gate/up) +
    // per-slot (down) input layouts.
    for &(e, n, k, t, topk, shared) in &[
        (8usize, 512usize, 2048usize, 1usize, 4usize, true),
        (8, 512, 2048, 1, 4, false),
        (16, 768, 2048, 3, 6, true),
    ] {
        let s = run_moe(&dev, e, n, k, t, topk, shared);
        println!("MoE IQ2_XXS [E{e} {n}x{k} t{t} topk{topk} shared{shared}]: max_abs={:.3e} max_rel={:.3e}", s.max_abs, s.max_rel);
        assert!(
            s.max_rel < 1e-3 && s.max_abs < 1e-1,
            "fused MoE diverged: max_abs={:.3e} max_rel={:.3e}",
            s.max_abs,
            s.max_rel
        );
    }
}
