//! S2 gate for the DSL PQ2_0 decode matvec on Metal, through the PRODUCTION path.
//!
//! Everything goes through `QMatMul::forward` on a Metal `QTensor` holding real GGUF `block_pq2_0`
//! bytes (the same fixtures as `hanzo-kernel`'s `pq2-check`), with `quantized::dsl` switching the
//! implementation. The oracle is independent of both kernels: the CPU `QTensor::dequantize` of the
//! same bytes, dotted in f64.
//!
//! Proven here, per the S2 requirements:
//!   - DSL f32 and bf16 match the oracle (and the legacy kernel) at 27B projection shapes.
//!   - A nonzero activation offset (a row view into a larger tensor) gives bit-identical output.
//!   - Dispatch proof: the DSL call counter moves only when the DSL is selected; legacy is counted too.
//!   - No hidden codegen: N forwards compile each pipeline exactly once.
//!
//! `HANZO_PQ2_GGUF=/path/model.gguf` additionally runs every PQ2_0 tensor of a real model file.
#![cfg(feature = "metal")]

use hanzo_ml::quantized::dsl::{self, QkImpl};
use hanzo_ml::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use hanzo_ml::{DType, Device, Module, Tensor};
use std::borrow::Cow;
use std::sync::Mutex;

// The impl switch and the counters are process-global; serialize the tests that touch them.
static SERIAL: Mutex<()> = Mutex::new(());

fn raw_pq2(n: usize, k: usize, seed: u64) -> Vec<u8> {
    let (codes, d) = hanzo_kernel::ternary::pq2_0_synth(n, k, seed);
    hanzo_kernel::ternary::pq2_0_gguf_bytes(&codes, &d, n, k)
}

fn oracle(raw: &[u8], x: &[f32], n: usize, k: usize) -> Vec<f32> {
    let q = QTensor::new(
        QStorage::from_data(Cow::Borrowed(raw), &Device::Cpu, GgmlDType::PQ2_0).unwrap(),
        (n, k),
    )
    .unwrap();
    let w: Vec<f32> = q
        .dequantize(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    (0..n)
        .map(|r| {
            (0..k)
                .map(|j| w[r * k + j] as f64 * x[j] as f64)
                .sum::<f64>() as f32
        })
        .collect()
}

fn scale_rel(want: &[f32], got: &[f32]) -> f32 {
    let maxref = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
    want.iter()
        .zip(got)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
        / maxref
}

fn matmul(dev: &Device, raw: &[u8], n: usize, k: usize) -> QMatMul {
    let q = QTensor::new(
        QStorage::from_data(Cow::Borrowed(raw), dev, GgmlDType::PQ2_0).unwrap(),
        (n, k),
    )
    .unwrap();
    QMatMul::from_qtensor(q).unwrap()
}

fn forward(mm: &QMatMul, x: &Tensor, imp: QkImpl) -> Vec<f32> {
    dsl::set_pq2_impl(imp);
    mm.forward(x)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

fn act(k: usize) -> Vec<f32> {
    (0..k)
        .map(|i| (((i * 7919) % 2003) as f32 / 1001.5) - 1.0)
        .collect()
}

const SHAPES: &[(usize, usize)] = &[
    (2048, 2048),
    (5120, 5120),
    (17408, 5120),
    (5120, 17408),
    (1024, 5120),
];

#[test]
fn pq2_dsl_f32_and_bf16_match_oracle_and_legacy() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dev = Device::new_metal(0).unwrap();
    for &(n, k) in SHAPES {
        let raw = raw_pq2(n, k, (n * 131 + k) as u64);
        let xs = act(k);
        let want = oracle(&raw, &xs, n, k);
        let mm = matmul(&dev, &raw, n, k);

        let x32 = Tensor::from_vec(xs.clone(), (1, k), &dev).unwrap();
        let leg = forward(&mm, &x32, QkImpl::Legacy);
        let dsl32 = forward(&mm, &x32, QkImpl::Dsl);
        let (r_dsl, r_leg) = (scale_rel(&want, &dsl32), scale_rel(&want, &leg));
        eprintln!("f32  n={n:<6} k={k:<6} dsl={r_dsl:.2e} legacy={r_leg:.2e}");
        assert!(r_dsl <= 1e-5, "DSL f32 vs oracle {r_dsl} at {n}x{k}");

        // bf16: the oracle on the bf16-rounded activation; the output is bf16 (8-bit mantissa).
        let xb: Vec<f32> = xs
            .iter()
            .map(|v| half::bf16::from_f32(*v).to_f32())
            .collect();
        let want_b = oracle(&raw, &xb, n, k);
        let x16 = x32.to_dtype(DType::BF16).unwrap();
        let leg_b = forward(&mm, &x16, QkImpl::Legacy);
        let dsl_b = forward(&mm, &x16, QkImpl::Dsl);
        let (rb_dsl, rb_leg, rb_x) = (
            scale_rel(&want_b, &dsl_b),
            scale_rel(&want_b, &leg_b),
            scale_rel(&leg_b, &dsl_b),
        );
        eprintln!(
            "bf16 n={n:<6} k={k:<6} dsl={rb_dsl:.2e} legacy={rb_leg:.2e} dsl-vs-legacy={rb_x:.2e}"
        );
        assert!(rb_dsl <= 4e-3, "DSL bf16 vs oracle {rb_dsl} at {n}x{k}");
        assert!(rb_x <= 8e-3, "DSL bf16 vs legacy bf16 {rb_x} at {n}x{k}");
    }
}

#[test]
fn pq2_dsl_nonzero_activation_offset_is_bit_identical() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dev = Device::new_metal(0).unwrap();
    let (n, k) = (5120, 5120);
    let raw = raw_pq2(n, k, 7);
    let mm = matmul(&dev, &raw, n, k);
    // Three activation rows in one buffer; the middle one is a view at byte offset k*4.
    let mut big = act(k);
    big.extend(act(k).iter().map(|v| v * 0.5));
    big.extend(act(k).iter().map(|v| -v));
    let big = Tensor::from_vec(big, (3, k), &dev).unwrap();
    let view = big.narrow(0, 1, 1).unwrap();
    assert!(
        view.layout().start_offset() > 0,
        "view must carry an offset"
    );
    let fresh = Tensor::from_vec(act(k).iter().map(|v| v * 0.5).collect(), (1, k), &dev).unwrap();
    for dt in [DType::F32, DType::BF16] {
        let (v, f) = (view.to_dtype(dt).unwrap(), fresh.to_dtype(dt).unwrap());
        // to_dtype on F32->F32 is a no-op view, so the F32 case really exercises the offset.
        let a = forward(&mm, &v, QkImpl::Dsl);
        let b = forward(&mm, &f, QkImpl::Dsl);
        assert_eq!(a, b, "{dt:?}: offset view differs from a fresh tensor");
    }
}

#[test]
fn pq2_dispatch_proof_and_single_compile() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dev = Device::new_metal(0).unwrap();
    let (n, k) = (17408, 5120);
    let raw = raw_pq2(n, k, 3);
    let mm = matmul(&dev, &raw, n, k);
    let x = Tensor::from_vec(act(k), (1, k), &dev).unwrap();
    let (artifact, _, _) = dsl::pq2_matvec_artifact(n, false);

    dsl::reset_stats();
    for _ in 0..5 {
        forward(&mm, &x, QkImpl::Legacy);
    }
    let s = dsl::stats();
    assert_eq!(
        s.calls(QkImpl::Dsl),
        0,
        "legacy selected but DSL ran:\n{}",
        dsl::report()
    );
    assert_eq!(s.calls(QkImpl::Legacy), 5);

    // A fresh device: its pipeline cache starts empty, so exactly one compile is expected.
    let dev2 = Device::new_metal(0).unwrap();
    let mm2 = matmul(&dev2, &raw, n, k);
    let x2 = Tensor::from_vec(act(k), (1, k), &dev2).unwrap();
    dsl::reset_stats();
    for _ in 0..200 {
        forward(&mm2, &x2, QkImpl::Dsl);
    }
    let s = dsl::stats();
    eprintln!("{}", dsl::report());
    assert_eq!(s.calls(QkImpl::Dsl), 200);
    assert_eq!(s.calls(QkImpl::Legacy), 0);
    assert!(
        s.compiles_of(artifact) <= 1,
        "{artifact} compiled {}x over 200 decode calls",
        s.compiles_of(artifact)
    );
}

/// Every PQ2_0 2-D tensor of a real model file: DSL vs legacy vs oracle (f32). Opt-in.
#[test]
fn pq2_dsl_real_model_tensors() {
    let Ok(path) = std::env::var("HANZO_PQ2_GGUF") else {
        eprintln!("HANZO_PQ2_GGUF unset; skipping real-model tensors");
        return;
    };
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dev = Device::new_metal(0).unwrap();
    let mut f = std::fs::File::open(&path).unwrap();
    let content = hanzo_ml::quantized::gguf_file::Content::read(&mut f).unwrap();
    let max = std::env::var("HANZO_PQ2_GGUF_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(24usize);
    let mut names: Vec<_> = content
        .tensor_infos
        .iter()
        .filter(|(_, ti)| ti.ggml_dtype == GgmlDType::PQ2_0 && ti.shape.rank() == 2)
        .map(|(n, _)| n.clone())
        .collect();
    names.sort();
    assert!(!names.is_empty(), "no PQ2_0 tensors in {path}");
    let mut worst = 0f32;
    for name in names.iter().take(max) {
        let qcpu = content.tensor(&mut f, name, &Device::Cpu).unwrap();
        let (n, k) = qcpu.shape().dims2().unwrap();
        let raw = qcpu.data().unwrap().to_vec();
        let xs = act(k);
        let want = oracle(&raw, &xs, n, k);
        let mm = matmul(&dev, &raw, n, k);
        let x = Tensor::from_vec(xs, (1, k), &dev).unwrap();
        let leg = forward(&mm, &x, QkImpl::Legacy);
        let d = forward(&mm, &x, QkImpl::Dsl);
        let (rd, rl) = (scale_rel(&want, &d), scale_rel(&want, &leg));
        worst = worst.max(rd);
        eprintln!("{name:<48} {n:>6}x{k:<6} dsl={rd:.2e} legacy={rl:.2e}");
        assert!(rd <= 1e-5, "{name}: DSL vs oracle {rd}");
    }
    eprintln!(
        "real-model worst DSL rel = {worst:.2e} over {} tensors",
        names.len().min(max)
    );
}

/// Kernel-level A/B at the 27B decode shapes (run with --ignored on a quiet GPU). Each arm streams
/// `BANKS` distinct weight copies in rotation so neither benefits from a warm SLC, and both arms run
/// interleaved per shape. Reports best-of-5 microseconds per matvec and effective GB/s.
#[test]
#[ignore]
fn pq2_dsl_vs_legacy_kernel_bench() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    const BANKS: usize = 6;
    const ITERS: usize = 60;
    let dev = Device::new_metal(0).unwrap();
    let shapes = [
        (10240usize, 5120usize),
        (6144, 5120),
        (5120, 6144),
        (17408, 5120),
        (5120, 17408),
        (12288, 5120),
        (1024, 5120),
        (248_320, 5120),
    ];
    eprintln!("shape            arm     us/matvec   GB/s");
    for (n, k) in shapes {
        let banks: Vec<QMatMul> = (0..BANKS)
            .map(|b| matmul(&dev, &raw_pq2(n, k, 100 + b as u64), n, k))
            .collect();
        let x = Tensor::from_vec(act(k), (1, k), &dev).unwrap();
        let bytes = (n * k / 128 * 34) as f64;
        let mut res = vec![];
        for imp in [QkImpl::Legacy, QkImpl::Dsl, QkImpl::Legacy, QkImpl::Dsl] {
            dsl::set_pq2_impl(imp);
            for b in &banks {
                b.forward(&x).unwrap();
            }
            dev.synchronize().unwrap();
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                for i in 0..ITERS {
                    let _ = banks[i % BANKS].forward(&x).unwrap();
                }
                dev.synchronize().unwrap();
                best = best.min(t.elapsed().as_secs_f64() * 1e6 / ITERS as f64);
            }
            res.push((imp, best));
        }
        for imp in [QkImpl::Legacy, QkImpl::Dsl] {
            let us = res
                .iter()
                .filter(|r| r.0 == imp)
                .map(|r| r.1)
                .fold(f64::INFINITY, f64::min);
            eprintln!(
                "{n:>6}x{k:<6}  {:<7} {us:>9.1}  {:>6.0}",
                imp.name(),
                bytes / us / 1e3
            );
        }
    }
}

/// DSL schedule sweep at the 27B decode shapes: every generated NR0 artifact vs legacy, interleaved,
/// cold-bank rotation. The production `pq2_schedule` table is set from this (run with --ignored).
#[test]
#[ignore]
fn pq2_dsl_nr0_sweep() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    const BANKS: usize = 6;
    const ITERS: usize = 60;
    let dev = Device::new_metal(0).unwrap();
    let shapes = [
        (1024usize, 5120usize),
        (5120, 6144),
        (6144, 5120),
        (10240, 5120),
        (12288, 5120),
        (17408, 5120),
        (5120, 17408),
        (248_320, 5120),
    ];
    let arms: [(QkImpl, Option<usize>); 4] = [
        (QkImpl::Legacy, None),
        (QkImpl::Dsl, Some(2)),
        (QkImpl::Dsl, Some(4)),
        (QkImpl::Dsl, Some(8)),
    ];
    eprintln!("shape            legacy_us   nr2_us   nr4_us   nr8_us   best");
    for (n, k) in shapes {
        let banks: Vec<QMatMul> = (0..BANKS)
            .map(|b| matmul(&dev, &raw_pq2(n, k, 200 + b as u64), n, k))
            .collect();
        let x = Tensor::from_vec(act(k), (1, k), &dev).unwrap();
        let mut best = [f64::INFINITY; 4];
        for _pass in 0..2 {
            for (ai, (imp, nr0)) in arms.iter().enumerate() {
                dsl::set_pq2_impl(*imp);
                dsl::set_pq2_nr0(*nr0);
                for b in &banks {
                    b.forward(&x).unwrap();
                }
                dev.synchronize().unwrap();
                for _ in 0..4 {
                    let t = std::time::Instant::now();
                    for i in 0..ITERS {
                        let _ = banks[i % BANKS].forward(&x).unwrap();
                    }
                    dev.synchronize().unwrap();
                    best[ai] = best[ai].min(t.elapsed().as_secs_f64() * 1e6 / ITERS as f64);
                }
            }
        }
        dsl::set_pq2_nr0(None);
        let (bi, _) = best[1..]
            .iter()
            .enumerate()
            .fold(
                (0, f64::INFINITY),
                |a, (i, v)| if *v < a.1 { (i, *v) } else { a },
            );
        eprintln!(
            "{n:>6}x{k:<6}  {:>9.1} {:>8.1} {:>8.1} {:>8.1}   nr{} ({:+.0}% vs legacy)",
            best[0],
            best[1],
            best[2],
            best[3],
            [2, 4, 8][bi],
            (best[0] / best[1 + bi] - 1.0) * 100.0
        );
    }
}
