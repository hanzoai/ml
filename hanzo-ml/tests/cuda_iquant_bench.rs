//! Decode-throughput micro-benchmark for the native CUDA i-quant mmvq path vs the dequantize-to-f32
//! fallback. Run it TWICE (the path is selected once per process from the env):
//!   cargo test -p hanzo-ml --features cuda --release --test cuda_iquant_bench -- --nocapture --ignored
//!   HANZO_IQ_DEQUANT_FALLBACK=1 cargo test ... (same line)
//! and compare the reported us/matvec + GB/s. Marked #[ignore] so it never runs in the normal suite.
#![cfg(feature = "cuda")]

use hanzo_ml::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use hanzo_ml::{Device, Module, Tensor};
use std::borrow::Cow;
use std::time::Instant;

fn pbyte(i: usize) -> u8 {
    let mut z = (i as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z >> 33) as u8
}

// Raw i-quant blocks (any byte pattern is a valid codebook block). IQ2_XXS = 66 B (f16 d + 64 qs);
// IQ4_XS = 136 B (f16 d + 134: u16 scales_h + 4 scales_l + 128 qs).
fn synth(dtype: GgmlDType, nout: usize, k: usize) -> Vec<u8> {
    use half::f16;
    let (d, nbytes) = match dtype {
        GgmlDType::IQ2_XXS => (0.05f32, 64usize),
        GgmlDType::IQ4_XS => (0.001, 134),
        _ => panic!("bench synth: {dtype:?} unsupported"),
    };
    let mut out = Vec::new();
    let mut c = 0usize;
    for _ in 0..nout * (k / 256) {
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        for _ in 0..nbytes {
            out.push(pbyte(c));
            c += 1;
        }
    }
    out
}

// m == 1 -> decode (mmvq dp4a); m > 8 -> prefill (qmmq int8-WMMA GEMM). The fallback path (set via
// HANZO_IQ_DEQUANT_FALLBACK=1) is the dequant-to-f32 round-trip for both.
fn bench_shape(dev: &Device, dtype: GgmlDType, nout: usize, k: usize, m: usize) -> hanzo_ml::Result<()> {
    let raw = synth(dtype, nout, k);
    let bytes = raw.len();
    let qs = QStorage::from_data(Cow::Owned(raw), dev, dtype)?;
    let q = QTensor::new(qs, (nout, k))?;
    let matmul = QMatMul::from_qtensor(q)?;
    let x = Tensor::from_vec(
        (0..m * k).map(|i| (pbyte(i) as f32 / 128.0) - 1.0).collect(),
        (m, k),
        dev,
    )?;

    // Warmup + sync.
    for _ in 0..10 {
        let _ = matmul.forward(&x)?;
    }
    dev.synchronize()?;

    let iters = if m > 8 { 100 } else { 300 };
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = matmul.forward(&x)?;
    }
    dev.synchronize()?;
    let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let fallback = std::env::var("HANZO_IQ_DEQUANT_FALLBACK").is_ok();
    let path = match (fallback, m) {
        (true, _) => "dequant-fallback",
        (false, 1) => "native-dp4a",
        (false, _) => "native-qmmq",
    };
    let kind = if m == 1 { "decode" } else { "prefill" };
    println!(
        "[{path:>16}] {kind} {dtype:?} [n={nout:>5} k={k:>5} m={m:>4}]  {us:9.2} us/call",
    );
    Ok(())
}

#[test]
#[ignore]
fn bench_iq2xxs_decode() {
    let dev = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip: no CUDA device ({e})");
            return;
        }
    };
    // Decode (m=1, mmvq) + prefill (m=128, qmmq) on realistic proj shapes (attn k=2048; FFN k=4096),
    // for IQ2_XXS (grid codebook) and IQ4_XS (LUT codebook -- the dominant type in UD-IQ2_M MoE quants).
    for dtype in [GgmlDType::IQ2_XXS, GgmlDType::IQ4_XS] {
        for &(n, k, m) in &[(4096usize, 4096usize, 1usize), (8192, 4096, 1), (4096, 4096, 128)] {
            bench_shape(&dev, dtype, n, k, m).unwrap();
        }
    }
}
