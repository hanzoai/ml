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

// IQ2_XXS raw blocks (66 B = f16 d + 64 B qs) -- any byte pattern is a valid codebook block.
fn synth_iq2xxs(nout: usize, k: usize) -> Vec<u8> {
    use half::f16;
    let mut out = Vec::new();
    let mut c = 0usize;
    for _ in 0..nout * (k / 256) {
        out.extend_from_slice(&f16::from_f32(0.05).to_le_bytes());
        for _ in 0..64 {
            out.push(pbyte(c));
            c += 1;
        }
    }
    out
}

fn bench_shape(dev: &Device, nout: usize, k: usize) -> hanzo_ml::Result<()> {
    let raw = synth_iq2xxs(nout, k);
    let bytes = raw.len();
    let qs = QStorage::from_data(Cow::Owned(raw), dev, GgmlDType::IQ2_XXS)?;
    let q = QTensor::new(qs, (nout, k))?;
    let matmul = QMatMul::from_qtensor(q)?;
    let x = Tensor::from_vec((0..k).map(|i| (pbyte(i) as f32 / 128.0) - 1.0).collect(), (1, k), dev)?;

    // Warmup + sync.
    for _ in 0..20 {
        let _ = matmul.forward(&x)?;
    }
    dev.synchronize()?;

    let iters = 300;
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = matmul.forward(&x)?;
    }
    dev.synchronize()?;
    let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let gbps = bytes as f64 / (us * 1e-6) / 1e9;
    let path = if std::env::var("HANZO_IQ_DEQUANT_FALLBACK").is_ok() {
        "dequant-fallback"
    } else {
        "native-dp4a"
    };
    println!(
        "[{path:>16}] IQ2_XXS [{nout:>5} x {k:>5}]  {us:8.2} us/matvec   {gbps:7.1} GB/s   ({:.2} MB weight)",
        bytes as f64 / 1e6
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
    // Realistic decode-proj shapes (attn k=2048; FFN k=4096; large lm_head-ish).
    for &(n, k) in &[(2048usize, 2048usize), (4096, 4096), (8192, 4096)] {
        bench_shape(&dev, n, k).unwrap();
    }
}
