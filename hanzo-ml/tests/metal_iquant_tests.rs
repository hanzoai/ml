//! Numerical validation of the native Metal i-quant kernels (kernel_mul_mv_iq*_f32 /
//! kernel_mul_mm_iq*_f32 in quantized.metal) against the CPU `to_float` reference, on a real Apple
//! GPU (M4 Max / dbc).
//!
//! Two gates:
//!   1. DEQUANT round-trip (`metal_dequant_*_bit_exact`): upload raw GGML block bytes to a Metal
//!      buffer, run `QTensor::dequantize` (CPU codebook decode of the resident bytes), and assert it
//!      is BIT-IDENTICAL to the CPU-side dequantize of the same bytes. Proves the block bytes survive
//!      the Metal buffer upload/readback byte-for-byte.
//!   2. MATVEC (`metal_matvec_*_matches_cpu`): load the bytes as a Metal QTensor and run
//!      `QMatMul::forward` with a [1, k] f32 activation -- routing through `fwd` -> the native
//!      `kernel_mul_mv_iq*_f32` matvec (which reconstructs the codebook weight to float in-kernel and
//!      does f32 MACs against the raw f32 activation). Compared to the CPU f64 matvec of the SAME
//!      dequantized weights over the SAME activation; the only residual is f32-vs-f64 accumulation
//!      order. A genuine codebook-decode bug (wrong grid index / sign / scale) spikes max_rel to
//!      O(1e-2..1).
//!
//! Skips cleanly when no Metal GPU is present. Requires the `metal` feature.
#![cfg(feature = "metal")]

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
// dequantized weights O(0.5). Block layouts are bit-for-bit the CPU block structs. (Mirrors the CUDA
// i-quant test's synth so the same scale tuning holds.)
fn synth(dtype: GgmlDType, nout: usize, k: usize) -> Vec<u8> {
    use half::f16;
    let mut out: Vec<u8> = Vec::new();
    let mut c = 0usize;
    let be = dtype.block_size();
    let nblk = nout * (k / be);
    let emit_d_then = |scale: f32, nbytes: usize, out: &mut Vec<u8>, c: &mut usize| {
        out.extend_from_slice(&f16::from_f32(pseudo(*c) * scale).to_le_bytes());
        *c += 1;
        for _ in 0..nbytes {
            out.push(pbyte(*c));
            *c += 1;
        }
    };
    match dtype {
        // f16 d + qs (block byte size minus the 2-byte d).
        GgmlDType::IQ2_XXS => (0..nblk).for_each(|_| emit_d_then(0.05, 64, &mut out, &mut c)),
        GgmlDType::IQ2_XS => (0..nblk).for_each(|_| emit_d_then(0.05, 72, &mut out, &mut c)),
        GgmlDType::IQ2_S => (0..nblk).for_each(|_| emit_d_then(0.05, 80, &mut out, &mut c)),
        GgmlDType::IQ3_XXS => (0..nblk).for_each(|_| emit_d_then(0.02, 96, &mut out, &mut c)),
        GgmlDType::IQ3_S => (0..nblk).for_each(|_| emit_d_then(0.004, 108, &mut out, &mut c)),
        GgmlDType::IQ1_S => (0..nblk).for_each(|_| emit_d_then(0.05, 48, &mut out, &mut c)),
        // block_iq4_xs (136 B): f16 d + u16 scales_h + scales_l[4] + qs[128]. dl = d*(ls-32) amplifies.
        GgmlDType::IQ4_XS => (0..nblk).for_each(|_| emit_d_then(0.001, 134, &mut out, &mut c)),
        // block_iq4_nl (18 B, 32 elems): f16 d + qs[16]. value = d*kvalues_iq4nl[nibble], |kv|<=127.
        GgmlDType::IQ4_NL => (0..nblk).for_each(|_| emit_d_then(0.005, 16, &mut out, &mut c)),
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
        // K-quants: quantize random f32 (candle's CPU quantizer) to valid blocks, then reuse the
        // identical bytes on both devices. block_q4_K = 144 B, block_q6_K = 210 B.
        GgmlDType::Q4K | GgmlDType::Q6K => {
            let w: Vec<f32> = (0..nblk * be).map(|i| pseudo(i) * 0.5).collect();
            let t = Tensor::from_vec(w, (nout, k), &Device::Cpu).unwrap();
            return QTensor::quantize(&t, dtype).unwrap().data().unwrap().into_owned();
        }
        _ => panic!("synth: {dtype:?} is not a wired Metal i-quant type"),
    }
    out
}

struct ErrStat {
    max_abs: f32,
    max_rel: f32,
}

// One (dtype, nout, k) decode case: native Metal matvec vs CPU f64 reference of the SAME dequantized
// weights (b_size = 1 -> fwd routes to fwd_mv -> kernel_mul_mv_iq*_f32).
fn run_case(dev: &Device, dtype: GgmlDType, nout: usize, k: usize) -> hanzo_ml::Result<ErrStat> {
    let cpu = Device::Cpu;
    let raw = synth(dtype, nout, k);

    // Ground-truth weights: dequantize the SAME raw bytes on the CPU (exactly what the kernel decodes).
    let q_cpu = QTensor::new(
        QStorage::from_data(Cow::Owned(raw.clone()), &cpu, dtype)?,
        (nout, k),
    )?;
    let w_deq: Vec<f32> = q_cpu.dequantize(&cpu)?.flatten_all()?.to_vec1::<f32>()?;

    // Native Metal decode: load the bytes on the GPU, forward a [1, k] f32 activation through QMatMul.
    let q_metal = QTensor::new(QStorage::from_data(Cow::Owned(raw), dev, dtype)?, (nout, k))?;
    let matmul = QMatMul::from_qtensor(q_metal)?;
    let x_host: Vec<f32> = (0..k).map(|i| pseudo(i + 1_000_003)).collect();
    let x = Tensor::from_vec(x_host.clone(), (1, k), dev)?;
    let y_metal: Vec<f32> = matmul.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(y_metal.len(), nout);

    // Reference: CPU f64 matvec of the dequantized weights over the same f32 activation. The Metal
    // kernel does the same reconstruction and dots in f32, so the residual is pure accumulation order.
    let mut max_abs = 0f32;
    let mut sse = 0f64;
    let mut ref_sq = 0f64;
    for n in 0..nout {
        let mut r = 0f64;
        for j in 0..k {
            r += w_deq[n * k + j] as f64 * x_host[j] as f64;
        }
        let g = y_metal[n] as f64;
        max_abs = max_abs.max((g - r).abs() as f32);
        sse += (g - r) * (g - r);
        ref_sq += r * r;
    }
    let ref_rms = (ref_sq / nout as f64).sqrt();
    let max_rel = ((sse / nout as f64).sqrt() / ref_rms.max(1e-9)) as f32;
    Ok(ErrStat { max_abs, max_rel })
}

fn check(dtype: GgmlDType) {
    let dev = match Device::new_metal(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip {dtype:?}: no Metal device ({e})");
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
        assert!(
            s.max_rel < 2e-3 && s.max_abs < 1e-1,
            "{dtype:?} [{nout}x{k}] native Metal decode diverged: max_abs={:.3e} max_rel={:.3e}",
            s.max_abs,
            s.max_rel
        );
    }
}

// DEQUANT round-trip: Metal QTensor::dequantize (CPU decode of the Metal-resident bytes) must be
// BIT-IDENTICAL to the CPU-side dequantize of the same bytes -- proves the buffer upload/readback is
// byte-exact for this block type.
fn check_dequant_bit_exact(dtype: GgmlDType) {
    let dev = match Device::new_metal(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip dequant {dtype:?}: no Metal device ({e})");
            return;
        }
    };
    let cpu = Device::Cpu;
    let (nout, k) = (256usize, 2048usize);
    let raw = synth(dtype, nout, k);
    let deq_cpu: Vec<f32> = QTensor::new(
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
    let deq_metal: Vec<f32> = QTensor::new(
        QStorage::from_data(Cow::Owned(raw), &dev, dtype).unwrap(),
        (nout, k),
    )
    .unwrap()
    .dequantize(&dev)
    .unwrap()
    .to_dtype(hanzo_ml::DType::F32)
    .unwrap()
    .flatten_all()
    .unwrap()
    .to_vec1::<f32>()
    .unwrap();
    assert_eq!(deq_cpu.len(), deq_metal.len());
    let mut max_abs = 0f32;
    for (a, b) in deq_cpu.iter().zip(&deq_metal) {
        max_abs = max_abs.max((a - b).abs());
    }
    println!("dequant {dtype:?}: bit-exact max_abs={max_abs:.3e}");
    assert_eq!(
        max_abs, 0.0,
        "{dtype:?} Metal dequantize not bit-identical to CPU: max_abs={max_abs:.3e}"
    );
}

// ---- MATVEC (native kernel_mul_mv_iq*_f32) vs CPU f64 reference ----
#[test]
fn metal_matvec_iq1s_matches_cpu() {
    check(GgmlDType::IQ1_S);
}
#[test]
fn metal_matvec_iq1m_matches_cpu() {
    check(GgmlDType::IQ1_M);
}
#[test]
fn metal_matvec_iq2xxs_matches_cpu() {
    check(GgmlDType::IQ2_XXS);
}
#[test]
fn metal_matvec_iq2xs_matches_cpu() {
    check(GgmlDType::IQ2_XS);
}
#[test]
fn metal_matvec_iq2s_matches_cpu() {
    check(GgmlDType::IQ2_S);
}
#[test]
fn metal_matvec_iq3xxs_matches_cpu() {
    check(GgmlDType::IQ3_XXS);
}
#[test]
fn metal_matvec_iq3s_matches_cpu() {
    check(GgmlDType::IQ3_S);
}
#[test]
fn metal_matvec_iq4xs_matches_cpu() {
    check(GgmlDType::IQ4_XS);
}
#[test]
fn metal_matvec_iq4nl_matches_cpu() {
    check(GgmlDType::IQ4_NL);
}
#[test]
fn metal_matvec_q4k_matches_cpu() {
    check(GgmlDType::Q4K);
}
#[test]
fn metal_matvec_q6k_matches_cpu() {
    check(GgmlDType::Q6K);
}

// A bf16 activation routes fwd_mv to the bf16-native matvec (bf16 src1/dst, f32 accumulation). It must
// be BIT-IDENTICAL to the f32 path run on the SAME bf16-rounded activation and rounded to bf16 -- which
// is exactly what GgufMatMul's f32 fallback computes (upcast bf16->f32, f32 matvec, round back to bf16).
// Both accumulate the same products in f32 and apply the same final bf16 rounding, so the result is exact.
fn check_bf16_matches_f32_rounded(dtype: GgmlDType) {
    let dev = match Device::new_metal(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip bf16 {dtype:?}: no Metal device ({e})");
            return;
        }
    };
    for &(nout, k) in &[(512usize, 2048usize), (768, 4096)] {
        let raw = synth(dtype, nout, k);
        let q = QTensor::new(QStorage::from_data(Cow::Owned(raw), &dev, dtype).unwrap(), (nout, k))
            .unwrap();
        let matmul = QMatMul::from_qtensor(q).unwrap();
        let x_host: Vec<f32> = (0..k).map(|i| pseudo(i + 1_000_003)).collect();
        // The activation as the model carries it: bf16. Upcast is exact, so the f32 path sees the
        // identical numeric input the bf16 kernel reads.
        let x_bf16 = Tensor::from_vec(x_host, (1, k), &dev)
            .unwrap()
            .to_dtype(hanzo_ml::DType::BF16)
            .unwrap();
        let x_f32 = x_bf16.to_dtype(hanzo_ml::DType::F32).unwrap();

        let y_bf16 = matmul.forward(&x_bf16).unwrap();
        assert_eq!(
            y_bf16.dtype(),
            hanzo_ml::DType::BF16,
            "{dtype:?} bf16 activation did not take the bf16-native matvec"
        );
        let y_bf16: Vec<f32> = y_bf16
            .to_dtype(hanzo_ml::DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let y_ref: Vec<f32> = matmul
            .forward(&x_f32)
            .unwrap()
            .to_dtype(hanzo_ml::DType::BF16)
            .unwrap()
            .to_dtype(hanzo_ml::DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let mut max_abs = 0f32;
        for (a, b) in y_bf16.iter().zip(&y_ref) {
            max_abs = max_abs.max((a - b).abs());
        }
        println!("bf16 {dtype:?} [{nout}x{k}]: max_abs vs f32-rounded={max_abs:.3e}");
        assert_eq!(
            max_abs, 0.0,
            "{dtype:?} [{nout}x{k}] bf16 matvec not bit-identical to f32-then-round-bf16: {max_abs:.3e}"
        );
    }
}
#[test]
fn metal_matvec_q4k_bf16_matches_f32() {
    check_bf16_matches_f32_rounded(GgmlDType::Q4K);
}
#[test]
fn metal_matvec_q6k_bf16_matches_f32() {
    check_bf16_matches_f32_rounded(GgmlDType::Q6K);
}

// ---- DEQUANT round-trip bit-exactness (upload/readback) ----
#[test]
fn metal_dequant_iq1s_bit_exact() {
    check_dequant_bit_exact(GgmlDType::IQ1_S);
}
#[test]
fn metal_dequant_iq2xxs_bit_exact() {
    check_dequant_bit_exact(GgmlDType::IQ2_XXS);
}
#[test]
fn metal_dequant_iq3s_bit_exact() {
    check_dequant_bit_exact(GgmlDType::IQ3_S);
}
#[test]
fn metal_dequant_iq4xs_bit_exact() {
    check_dequant_bit_exact(GgmlDType::IQ4_XS);
}

// ---- PREFILL (native kernel_mul_mm_iq*_f32, m>1 -> fwd routes to the mm GEMM). f16 tile accumulation
// -> a looser tolerance than the f32 decode path. Covers the primary IQ1_S + a representative grid
// (IQ2_XXS) + LUT (IQ4_XS). ----
fn run_prefill(dev: &Device, dtype: GgmlDType, nout: usize, k: usize, m: usize) -> ErrStat {
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
    let x_host: Vec<f32> = (0..m * k).map(|i| pseudo(i + 555)).collect();
    let x = Tensor::from_vec(x_host.clone(), (m, k), dev).unwrap();
    let y: Vec<f32> = matmul
        .forward(&x)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let mut max_abs = 0f32;
    let mut sse = 0f64;
    let mut rs = 0f64;
    for mi in 0..m {
        for n in 0..nout {
            let mut r = 0f64;
            for j in 0..k {
                r += w_deq[n * k + j] as f64 * x_host[mi * k + j] as f64;
            }
            let g = y[mi * nout + n] as f64;
            max_abs = max_abs.max((g - r).abs() as f32);
            sse += (g - r) * (g - r);
            rs += r * r;
        }
    }
    let cnt = (m * nout) as f64;
    let max_rel = ((sse / cnt).sqrt() / (rs / cnt).sqrt().max(1e-9)) as f32;
    ErrStat { max_abs, max_rel }
}

#[test]
fn metal_prefill_iquant_matches_cpu() {
    let dev = match Device::new_metal(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip prefill: no Metal device ({e})");
            return;
        }
    };
    for dtype in [GgmlDType::IQ1_S, GgmlDType::IQ2_XXS, GgmlDType::IQ4_XS] {
        for &m in &[16usize, 32] {
            let s = run_prefill(&dev, dtype, 512, 2048, m);
            println!(
                "prefill {dtype:?} m={m}: max_abs={:.3e} max_rel={:.3e}",
                s.max_abs, s.max_rel
            );
            // f16-tile mm accumulation -> ~1e-2 tolerance (matches the k-quant Metal mm path).
            assert!(
                s.max_rel < 3e-2,
                "{dtype:?} Metal prefill m={m} diverged: max_abs={:.3e} max_rel={:.3e}",
                s.max_abs,
                s.max_rel
            );
        }
    }
}

// ---- INDEXED (fused) MoE. Builds an [E, n, k] quantized expert bank and routes t tokens (topk
// distinct experts each) through `QTensor::indexed_moe_forward`. The `t` value selects the SAME path
// production does: t == 1 (decode) dispatches `call_mul_mv_id` (per-slot matrix-VECTOR), t > 1
// (prefill) dispatches `call_mul_mm_id` (expert-grouped GEMM). Reference = CPU f64 per-(token,slot)
// matvec of the dequantized routed expert. `per_slot` picks the down-proj input layout ([t,topk,k],
// one row per slot) vs the shared gate/up layout ([t,1,k]). ----
#[allow(clippy::too_many_arguments)]
fn run_moe_indexed(
    dev: &Device,
    dtype: GgmlDType,
    e_cnt: usize,
    n: usize,
    k: usize,
    t: usize,
    topk: usize,
    per_slot: bool,
) -> ErrStat {
    let cpu = Device::Cpu;
    // One synth over E*n rows = E distinct [n,k] expert weight banks (the counter advances per row).
    let raw = synth(dtype, e_cnt * n, k);
    let w_deq: Vec<f32> = QTensor::new(
        QStorage::from_data(Cow::Owned(raw.clone()), &cpu, dtype).unwrap(),
        (e_cnt, n, k),
    )
    .unwrap()
    .dequantize(&cpu)
    .unwrap()
    .flatten_all()
    .unwrap()
    .to_vec1::<f32>()
    .unwrap();
    let bank = QTensor::new(
        QStorage::from_data(Cow::Owned(raw), dev, dtype).unwrap(),
        (e_cnt, n, k),
    )
    .unwrap();

    // Router: topk distinct experts per token (topk <= e_cnt), varied per token.
    let mut ids_host = vec![0u32; t * topk];
    for ti in 0..t {
        let base = (ti * 3) % e_cnt;
        for j in 0..topk {
            ids_host[ti * topk + j] = ((base + j) % e_cnt) as u32;
        }
    }
    let ids = Tensor::from_vec(ids_host.clone(), (t, topk), dev).unwrap();

    // gate/up share one input row per token (s=1); down feeds a distinct row per slot (s=topk).
    let s = if per_slot { topk } else { 1 };
    let x_host: Vec<f32> = (0..t * s * k).map(|i| pseudo(i + 777)).collect();
    let x = Tensor::from_vec(x_host.clone(), (t, s, k), dev).unwrap();

    let y: Vec<f32> = bank
        .indexed_moe_forward(&x, &ids)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(y.len(), t * topk * n);

    let mut max_abs = 0f32;
    let mut sse = 0f64;
    let mut rs = 0f64;
    for ti in 0..t {
        for j in 0..topk {
            let e = ids_host[ti * topk + j] as usize;
            let xrow = if per_slot { ti * topk + j } else { ti };
            for ni in 0..n {
                let mut r = 0f64;
                for kk in 0..k {
                    r += w_deq[(e * n + ni) * k + kk] as f64 * x_host[xrow * k + kk] as f64;
                }
                let g = y[(ti * topk + j) * n + ni] as f64;
                max_abs = max_abs.max((g - r).abs() as f32);
                sse += (g - r) * (g - r);
                rs += r * r;
            }
        }
    }
    let cnt = (t * topk * n) as f64;
    let max_rel = ((sse / cnt).sqrt() / (rs / cnt).sqrt().max(1e-9)) as f32;
    ErrStat { max_abs, max_rel }
}

#[test]
fn metal_moe_prefill_iquant_matches_cpu() {
    let dev = match Device::new_metal(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip moe prefill: no Metal device ({e})");
            return;
        }
    };
    // (dtype, E, n, k, t, topk, per_slot). E=2/t=40 => 40 rows/expert = 2 column tiles (BN=32); t=33
    // => 2 token tiles; per_slot exercises the down-proj [t,topk,k] input. IQ1_S primary + IQ2/IQ3.
    let cases = [
        (GgmlDType::IQ1_S, 2usize, 256usize, 2048usize, 40usize, 2usize, false),
        (GgmlDType::IQ1_S, 8, 256, 2048, 16, 2, true),
        (GgmlDType::IQ2_XXS, 4, 256, 2048, 33, 2, false),
        (GgmlDType::IQ3_S, 4, 512, 4096, 16, 2, false),
    ];
    for (dtype, e, n, k, t, topk, per_slot) in cases {
        let stat = run_moe_indexed(&dev, dtype, e, n, k, t, topk, per_slot);
        println!(
            "moe prefill {dtype:?} E={e} [{n}x{k}] t={t} topk={topk} per_slot={per_slot}: \
             max_abs={:.3e} max_rel={:.3e}",
            stat.max_abs, stat.max_rel
        );
        assert!(
            stat.max_rel < 3e-2 && stat.max_abs.is_finite(),
            "{dtype:?} Metal MoE prefill diverged: max_abs={:.3e} max_rel={:.3e}",
            stat.max_abs,
            stat.max_rel
        );
    }
}

// ---- FUSED MoE DECODE (native kernel_mul_mv_id_iq*_f32, t == 1 -> the per-slot matrix-VECTOR). This
// is the decode twin of the prefill test above and the path GLM-5.2 (UD-IQ1_S) takes for every
// generated token. `indexed_moe_forward` with t == 1 dispatches `call_mul_mv_id`, reading each routed
// slot's expert id on-device and dotting the resident IQ codebook weight against the f32 activation
// in ONE fused GPU dispatch -- replacing the per-expert single-core CPU-scalar fallback. Decode does
// pure f32 MACs (no f16 GEMM tile), so the tolerance is TIGHT (the f32-accumulation-order floor),
// matching the plain-matvec gate -- a codebook/index/scale bug spikes max_rel to O(1e-2..1). ----
#[test]
fn metal_moe_decode_iquant_matches_cpu() {
    let dev = match Device::new_metal(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip moe decode: no Metal device ({e})");
            return;
        }
    };
    // (dtype, E, n, k, topk, per_slot). t == 1 (decode). GLM-5.2 primary IQ1_S + a ksigns grid
    // (IQ2_XXS), a raw-byte-sign grid (IQ2_S), a 3-bit grid (IQ3_S), and the LUT (IQ4_XS). Both the
    // shared gate/up input (per_slot=false, s=1) and the per-slot down input (per_slot=true, s=topk)
    // routing layouts; attn-proj k=2048 and ffn k=4096; topk 6/8 (>1 slot/token, distinct experts).
    let cases = [
        (GgmlDType::IQ1_S, 16usize, 256usize, 2048usize, 8usize, false),
        (GgmlDType::IQ1_S, 16, 512, 4096, 8, true),
        (GgmlDType::IQ2_XXS, 12, 256, 2048, 6, false),
        (GgmlDType::IQ2_S, 8, 256, 2048, 6, true),
        (GgmlDType::IQ3_S, 8, 512, 4096, 6, false),
        (GgmlDType::IQ4_XS, 8, 256, 2048, 6, true),
    ];
    for (dtype, e, n, k, topk, per_slot) in cases {
        let stat = run_moe_indexed(&dev, dtype, e, n, k, 1, topk, per_slot);
        println!(
            "moe decode {dtype:?} E={e} [{n}x{k}] topk={topk} per_slot={per_slot}: \
             max_abs={:.3e} max_rel={:.3e}",
            stat.max_abs, stat.max_rel
        );
        assert!(
            stat.max_rel < 2e-3 && stat.max_abs < 1e-1,
            "{dtype:?} Metal MoE decode (mul_mv_id) diverged: max_abs={:.3e} max_rel={:.3e}",
            stat.max_abs,
            stat.max_rel
        );
    }
}

// ---- DECODE THROUGHPUT: the whole point of the kernel. A GLM-5.2-shaped IQ1_S expert bank routed for
// one token, timed on the Metal GPU (fused `call_mul_mv_id`) vs the CPU per-expert fallback (the
// single-core-scalar path that pegs one core to 99.9% and collapses ring decode). Proves the decode
// MoE compute is now ON THE GPU and reports the speedup + a per-decode-step budget. `#[ignore]` (a
// benchmark, not a gate); run with `--ignored --nocapture`. ----
#[test]
#[ignore]
fn metal_moe_decode_iquant_throughput() {
    let dev = match Device::new_metal(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip moe decode throughput: no Metal device ({e})");
            return;
        }
    };
    let cpu = Device::Cpu;
    // GLM-5.2-representative MoE FFN projection: n = moe-intermediate 1536, k = hidden 5120 (both
    // multiples of the IQ1_S 256 super-block), topk = 8 routed experts / token. E kept modest -- decode
    // cost is topk*n*k (routed slots), independent of the resident expert count.
    let (dtype, e_cnt, n, k, topk) = (GgmlDType::IQ1_S, 32usize, 1536usize, 5120usize, 8usize);
    let raw = synth(dtype, e_cnt * n, k);
    let bank_metal = QTensor::new(
        QStorage::from_data(Cow::Owned(raw.clone()), &dev, dtype).unwrap(),
        (e_cnt, n, k),
    )
    .unwrap();
    let bank_cpu = QTensor::new(
        QStorage::from_data(Cow::Owned(raw), &cpu, dtype).unwrap(),
        (e_cnt, n, k),
    )
    .unwrap();

    // One decode token, topk distinct experts (shared gate/up input, s = 1).
    let ids_host: Vec<u32> = (0..topk).map(|j| (j % e_cnt) as u32).collect();
    let x_host: Vec<f32> = (0..k).map(|i| pseudo(i + 42)).collect();
    let mk = |d: &Device| {
        (
            Tensor::from_vec(ids_host.clone(), (1, topk), d).unwrap(),
            Tensor::from_vec(x_host.clone(), (1, 1, k), d).unwrap(),
        )
    };
    let (ids_m, x_m) = mk(&dev);
    let (ids_c, x_c) = mk(&cpu);

    // Sanity: GPU decode == CPU decode (same fused-vs-fallback math) before timing.
    let ym: Vec<f32> = bank_metal
        .indexed_moe_forward(&x_m, &ids_m)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let yc: Vec<f32> = bank_cpu
        .indexed_moe_forward(&x_c, &ids_c)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let mut mrel = 0f64;
    let mut rs = 0f64;
    for (a, b) in ym.iter().zip(&yc) {
        mrel += (*a as f64 - *b as f64).powi(2);
        rs += (*b as f64).powi(2);
    }
    let mrel = (mrel / rs.max(1e-12)).sqrt();
    println!("gpu-vs-cpu decode rel={mrel:.3e} (topk={topk} n={n} k={k})");
    // ~q8 floor (1/256 ~= 3.9e-3): the CPU IQ1_S fallback quantizes the ACTIVATION to q8_K (integer
    // vec_dot), while the Metal kernel dots the RAW f32 activation -- the Metal path is the MORE precise
    // one (f64-exact at ~2e-7 in metal_moe_decode_iquant_matches_cpu). So GPU and CPU agree only to the
    // CPU's own activation-quant floor; this is a liveness sanity, not the correctness gate.
    assert!(mrel < 1.5e-2, "GPU decode disagrees with CPU beyond the q8 floor: rel={mrel:.3e}");

    let time_it = |run: &dyn Fn() -> Vec<f32>, iters: usize| -> f64 {
        let _ = run(); // warm
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            std::hint::black_box(run());
        }
        t0.elapsed().as_secs_f64() * 1e6 / iters as f64 // us / call
    };
    // Reading the output back to host forces the Metal command buffer to complete each iteration.
    let gpu = time_it(
        &|| {
            bank_metal
                .indexed_moe_forward(&x_m, &ids_m)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        },
        100,
    );
    let cpu_us = time_it(
        &|| {
            bank_cpu
                .indexed_moe_forward(&x_c, &ids_c)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        },
        20,
    );

    // A GLM-5.2 decode token drives ~3 such routed MoE projections (gate + up + down) per MoE layer.
    let proj_per_tok = 3.0;
    println!(
        "MoE decode (IQ1_S, topk={topk}, [{n}x{k}]): GPU {gpu:.1} us/proj | CPU {cpu_us:.1} us/proj \
         | speedup {:.1}x",
        cpu_us / gpu
    );
    println!(
        "per-MoE-layer decode (x{proj_per_tok} proj): GPU {:.1} us | CPU {:.1} us  \
         (GPU offloads the pegged core)",
        gpu * proj_per_tok,
        cpu_us * proj_per_tok
    );
    assert!(
        gpu < cpu_us,
        "GPU decode ({gpu:.1} us) is not faster than CPU fallback ({cpu_us:.1} us) -- kernel not offloading"
    );
}
