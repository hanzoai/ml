#![cfg(feature = "metal")]

use hanzo_kernel::ternary::{
    ptq1_0_gguf_bytes, ptq1_0_gguf_gemm_ref, ptq1_0_gguf_ref, ptq1_0_synth,
};
use hanzo_ml::quantized::dsl::{self, QkImpl};
use hanzo_ml::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use hanzo_ml::{DType, Device, Module, Tensor};
use std::borrow::Cow;
use std::sync::Mutex;

static SERIAL: Mutex<()> = Mutex::new(());

fn raw_ptq1(n: usize, k: usize, seed: u64) -> Vec<u8> {
    let (codes, d) = ptq1_0_synth(n, k, seed);
    ptq1_0_gguf_bytes(&codes, &d, n, k)
}

fn bench_raw_ptq1(n: usize, k: usize, seed: u64) -> Vec<u8> {
    let tile = n.min(256);
    let pat = raw_ptq1(tile, k, seed);
    let row_bytes = (k / 128) * 28;
    let mut out = Vec::with_capacity(n * row_bytes);
    while out.len() < n * row_bytes {
        let take = (n * row_bytes - out.len()).min(pat.len());
        out.extend_from_slice(&pat[..take]);
    }
    out
}

fn matmul(dev: &Device, raw: &[u8], n: usize, k: usize) -> QMatMul {
    let qt = QTensor::new(
        QStorage::from_data(Cow::Borrowed(raw), dev, GgmlDType::PTQ1_0).unwrap(),
        (n, k),
    )
    .unwrap();
    QMatMul::from_qtensor(qt).unwrap()
}

fn act(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 73 + 19) % 199) as f32 - 99.0) * 0.015625)
        .collect()
}

fn forward_ptq1(mm: &QMatMul, x: &Tensor, imp: QkImpl) -> Vec<f32> {
    dsl::set_ptq1_impl(imp);
    let y = mm.forward(x).unwrap().to_dtype(DType::F32).unwrap();
    y.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn scale_rel(want: &[f32], got: &[f32]) -> f32 {
    assert_eq!(want.len(), got.len());
    let scale = want
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max)
        .max(1e-6);
    want.iter()
        .zip(got.iter())
        .map(|(a, b)| (a - b).abs() / scale)
        .fold(0.0f32, f32::max)
}

fn dequant_matrix_oracle(raw: &[u8], n: usize, k: usize) -> Vec<f32> {
    let qt = QTensor::new(
        QStorage::from_data(Cow::Borrowed(raw), &Device::Cpu, GgmlDType::PTQ1_0).unwrap(),
        (n, k),
    )
    .unwrap();
    qt.dequantize(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

#[test]
fn ptq1_dsl_decode_parity_f32_and_bf16() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dev = Device::new_metal(0).unwrap();
    const SHAPES: &[(usize, usize)] = &[
        (1, 128),
        (3, 128),
        (7, 384),
        (257, 1152),
        (128, 128),
        (1280, 1280),
        (5120, 5120),
        (17408, 5120),
        (5120, 17408),
        (10240, 5120),
    ];

    // Cross-check hanzo-kernel PTQ1 reference against hanzo-ml QTensor::dequantize on a small matrix.
    let (sn, sk) = (7usize, 384usize);
    let sraw = raw_ptq1(sn, sk, 42);
    let sact = act(sk);
    let w_deq = dequant_matrix_oracle(&sraw, sn, sk);
    let ref_from_deq: Vec<f32> = (0..sn)
        .map(|r| {
            w_deq[r * sk..(r + 1) * sk]
                .iter()
                .zip(sact.iter())
                .map(|(w, x)| (*w as f64) * (*x as f64))
                .sum::<f64>() as f32
        })
        .collect();
    let ref_kernel = ptq1_0_gguf_ref(&sraw, &sact, sn, sk);
    assert!(
        scale_rel(&ref_from_deq, &ref_kernel) <= 1e-6,
        "QTensor::dequantize vs ptq1_0_gguf_ref mismatch"
    );

    for &(n, k) in SHAPES {
        let raw = raw_ptq1(n, k, (n * 131 + k) as u64);
        let xs = act(k);
        let want = ptq1_0_gguf_ref(&raw, &xs, n, k);
        let mm = matmul(&dev, &raw, n, k);

        let x32 = Tensor::from_vec(xs.clone(), (1, k), &dev).unwrap();
        let leg = forward_ptq1(&mm, &x32, QkImpl::Legacy);
        let dsl32 = forward_ptq1(&mm, &x32, QkImpl::Dsl);
        let (r_dsl, r_leg) = (scale_rel(&want, &dsl32), scale_rel(&want, &leg));
        eprintln!("ptq1 f32  n={n:<6} k={k:<6} dsl={r_dsl:.2e} legacy={r_leg:.2e}");
        assert!(r_dsl <= 1e-5, "DSL f32 vs oracle {r_dsl} at {n}x{k}");

        let xb: Vec<f32> = xs
            .iter()
            .map(|v| half::bf16::from_f32(*v).to_f32())
            .collect();
        let want_b = ptq1_0_gguf_ref(&raw, &xb, n, k);
        let x16 = x32.to_dtype(DType::BF16).unwrap();
        let leg_b = forward_ptq1(&mm, &x16, QkImpl::Legacy);
        let dsl_b = forward_ptq1(&mm, &x16, QkImpl::Dsl);
        let (rb_dsl, rb_leg, rb_x) = (
            scale_rel(&want_b, &dsl_b),
            scale_rel(&want_b, &leg_b),
            scale_rel(&leg_b, &dsl_b),
        );
        eprintln!(
            "ptq1 bf16 n={n:<6} k={k:<6} dsl={rb_dsl:.2e} legacy={rb_leg:.2e} dsl-vs-legacy={rb_x:.2e}"
        );
        assert!(rb_dsl <= 4e-3, "DSL bf16 vs oracle {rb_dsl} at {n}x{k}");
        assert!(rb_x <= 8e-3, "DSL bf16 vs legacy bf16 {rb_x} at {n}x{k}");
    }
    dsl::set_ptq1_impl(QkImpl::Legacy);
}

#[test]
fn ptq1_dsl_nonzero_activation_offset_and_gemm_prefill_parity() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dev = Device::new_metal(0).unwrap();

    // Nonzero activation offset view parity (M=1).
    let (n, k) = (5120usize, 5120usize);
    let raw = raw_ptq1(n, k, 7);
    let mm = matmul(&dev, &raw, n, k);
    let mut big = act(k);
    big.extend(act(k).iter().map(|v| v * 0.5));
    big.extend(act(k).iter().map(|v| -v));
    let big = Tensor::from_vec(big, (3, k), &dev).unwrap();
    let view = big.narrow(0, 1, 1).unwrap();
    assert!(view.layout().start_offset() > 0);
    let fresh = Tensor::from_vec(act(k).iter().map(|v| v * 0.5).collect(), (1, k), &dev).unwrap();
    for dt in [DType::F32, DType::BF16] {
        let a = forward_ptq1(&mm, &view.to_dtype(dt).unwrap(), QkImpl::Dsl);
        let b = forward_ptq1(&mm, &fresh.to_dtype(dt).unwrap(), QkImpl::Dsl);
        assert_eq!(a, b, "{dt:?}: offset view differs from fresh tensor");
    }

    // 3D batch [2, 11, k] + GEMM shapes spanning bm32 (m<=64) and bm64 (m>64).
    const CASES: &[(usize, usize, usize)] = &[
        (2, 5120, 5120),
        (17, 1024, 5120),
        (33, 5120, 5120),
        (64, 5120, 5120),
        (128, 1024, 5120),
        (512, 6144, 5120),
    ];
    for &(m, cn, ck) in CASES {
        let craw = raw_ptq1(cn, ck, (m * 31 + cn * 7 + ck) as u64);
        let cmm = matmul(&dev, &craw, cn, ck);
        let xs = act(m * ck);
        let want32 = ptq1_0_gguf_gemm_ref(&craw, &xs, m, cn, ck);
        let x32 = Tensor::from_vec(xs.clone(), (m, ck), &dev).unwrap();
        let leg32 = forward_ptq1(&cmm, &x32, QkImpl::Legacy);
        let dsl32 = forward_ptq1(&cmm, &x32, QkImpl::Dsl);
        let r32 = scale_rel(&want32, &dsl32);
        let rx32 = scale_rel(&leg32, &dsl32);
        eprintln!("ptq1 gemm f32  m={m:<4} {cn}x{ck} rel={r32:.2e} vs_legacy={rx32:.2e}");
        assert!(r32 <= 5e-4, "GEMM f32 m={m} {cn}x{ck} rel={r32}");

        let xb: Vec<f32> = xs
            .iter()
            .map(|v| half::bf16::from_f32(*v).to_f32())
            .collect();
        let want16 = ptq1_0_gguf_gemm_ref(&craw, &xb, m, cn, ck);
        let x16 = x32.to_dtype(DType::BF16).unwrap();
        let leg16 = forward_ptq1(&cmm, &x16, QkImpl::Legacy);
        let dsl16 = forward_ptq1(&cmm, &x16, QkImpl::Dsl);
        let r16 = scale_rel(&want16, &dsl16);
        let rx16 = scale_rel(&leg16, &dsl16);
        eprintln!("ptq1 gemm bf16 m={m:<4} {cn}x{ck} rel={r16:.2e} vs_legacy={rx16:.2e}");
        assert!(r16 <= 8e-3, "GEMM bf16 m={m} {cn}x{ck} rel={r16}");
    }

    // 3D batch tensor shape [2, 11, ck].
    let (b, t, ck, cn) = (2usize, 11usize, 5120usize, 1024usize);
    let braw = raw_ptq1(cn, ck, 99);
    let bmm = matmul(&dev, &braw, cn, ck);
    let bxs = act(b * t * ck);
    let bwant = ptq1_0_gguf_gemm_ref(&braw, &bxs, b * t, cn, ck);
    let bx = Tensor::from_vec(bxs, (b, t, ck), &dev).unwrap();
    let y = bmm.forward(&bx).unwrap();
    assert_eq!(y.dims(), &[b, t, cn]);
    let bgot = forward_ptq1(&bmm, &bx, QkImpl::Dsl);
    assert!(scale_rel(&bwant, &bgot) <= 5e-4, "3D batch GEMM mismatch");

    dsl::set_ptq1_impl(QkImpl::Legacy);
}

#[test]
fn ptq1_dispatch_proof_and_single_compile() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dev = Device::new_metal(0).unwrap();
    let (n, k) = (17408usize, 5120usize);
    let raw = raw_ptq1(n, k, 3);
    let mm = matmul(&dev, &raw, n, k);
    let xm1 = Tensor::from_vec(act(k), (1, k), &dev).unwrap();
    let xm33 = Tensor::from_vec(act(33 * k), (33, k), &dev).unwrap();
    let (mv_art, _, _) = dsl::ptq1_matvec_artifact(n, false);
    let (mm_art, _) = dsl::ptq1_matmul_artifact(33, n, false);

    dsl::reset_stats();
    for _ in 0..5 {
        forward_ptq1(&mm, &xm1, QkImpl::Legacy);
    }
    forward_ptq1(&mm, &xm33, QkImpl::Legacy);
    let s = dsl::stats();
    assert_eq!(s.calls(QkImpl::Dsl), 0);
    assert_eq!(s.calls(QkImpl::Legacy), 6);

    let dev2 = Device::new_metal(0).unwrap();
    let mm2 = matmul(&dev2, &raw, n, k);
    let xm1_2 = Tensor::from_vec(act(k), (1, k), &dev2).unwrap();
    let xm33_2 = Tensor::from_vec(act(33 * k), (33, k), &dev2).unwrap();
    dsl::reset_stats();
    for _ in 0..100 {
        forward_ptq1(&mm2, &xm1_2, QkImpl::Dsl);
    }
    for _ in 0..10 {
        forward_ptq1(&mm2, &xm33_2, QkImpl::Dsl);
    }
    let s = dsl::stats();
    eprintln!("{}", dsl::report());
    assert_eq!(s.calls(QkImpl::Dsl), 110);
    assert_eq!(s.calls(QkImpl::Legacy), 0);
    assert!(s.compiles_of(mv_art) <= 1);
    assert!(s.compiles_of(mm_art) <= 1);
    dsl::set_ptq1_impl(QkImpl::Legacy);
}

#[test]
fn ptq1_dsl_real_model_tensors() {
    let Ok(path) = hanzo_ml::utils::ml_env_var("ML_PTQ1_GGUF", "HANZO_PTQ1_GGUF") else {
        eprintln!("ML_PTQ1_GGUF unset; skipping real-model tensors");
        return;
    };
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dev = Device::new_metal(0).unwrap();
    let mut f = std::fs::File::open(&path).unwrap();
    let content = hanzo_ml::quantized::gguf_file::Content::read(&mut f).unwrap();
    let max = hanzo_ml::utils::ml_env_var("ML_PTQ1_GGUF_MAX", "HANZO_PTQ1_GGUF_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(24usize);
    let mut names: Vec<_> = content
        .tensor_infos
        .iter()
        .filter(|(_, ti)| ti.ggml_dtype == GgmlDType::PTQ1_0 && ti.shape.rank() == 2)
        .map(|(n, _)| n.clone())
        .collect();
    names.sort();
    assert!(!names.is_empty(), "no PTQ1_0 tensors in {path}");
    let mut worst_mv = 0f32;
    let mut worst_mm = 0f32;
    for name in names.iter().take(max) {
        let qcpu = content.tensor(&mut f, name, &Device::Cpu).unwrap();
        let (n, k) = qcpu.shape().dims2().unwrap();
        let raw = qcpu.data().unwrap().to_vec();
        let xs = act(k);
        let want = ptq1_0_gguf_ref(&raw, &xs, n, k);
        let mm = matmul(&dev, &raw, n, k);
        let x1 = Tensor::from_vec(xs, (1, k), &dev).unwrap();
        let d1 = forward_ptq1(&mm, &x1, QkImpl::Dsl);
        let r1 = scale_rel(&want, &d1);
        worst_mv = worst_mv.max(r1);

        let x33s = act(33 * k);
        let want33 = ptq1_0_gguf_gemm_ref(&raw, &x33s, 33, n, k);
        let x33 = Tensor::from_vec(x33s, (33, k), &dev).unwrap();
        let d33 = forward_ptq1(&mm, &x33, QkImpl::Dsl);
        let r33 = scale_rel(&want33, &d33);
        worst_mm = worst_mm.max(r33);

        eprintln!("{name:<48} {n:>6}x{k:<6} mv={r1:.2e} mm33={r33:.2e}");
        assert!(r1 <= 1e-5, "{name}: M=1 rel={r1}");
        assert!(r33 <= 5e-4, "{name}: M=33 rel={r33}");
    }
    eprintln!(
        "real-model worst PTQ1 rel: mv={worst_mv:.2e} mm33={worst_mm:.2e} over {} tensors",
        names.len().min(max)
    );
    dsl::set_ptq1_impl(QkImpl::Legacy);
}

#[test]
#[ignore]
fn ptq1_schedule_and_kernel_ab_bench() {
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

    eprintln!("=== PTQ1_0 (nsg, nr0) decode sweep (BANKS={BANKS}, F32 & BF16) ===");
    for (n, k) in shapes {
        let banks: Vec<QMatMul> = (0..BANKS)
            .map(|b| matmul(&dev, &bench_raw_ptq1(n, k, 200 + b as u64), n, k))
            .collect();
        let x32 = Tensor::from_vec(act(k), (1, k), &dev).unwrap();
        for dt in [DType::F32, DType::BF16] {
            let x = x32.to_dtype(dt).unwrap();
            let time_arm = |imp: QkImpl| -> f64 {
                dsl::set_ptq1_impl(imp);
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
                best
            };

            dsl::set_pq2_nsg(None);
            dsl::set_pq2_nr0(None);
            let leg_us = time_arm(QkImpl::Legacy);
            let mut row = format!("{n:>6}x{k:<5} {dt:?} leg={leg_us:6.1}us");
            let mut best_cfg = ((2, 4), f64::INFINITY);
            for (nsg, nr0) in [(2, 2), (2, 4), (2, 8), (4, 2), (4, 4), (4, 8)] {
                dsl::set_pq2_nsg(Some(nsg));
                dsl::set_pq2_nr0(Some(nr0));
                let us = time_arm(QkImpl::Dsl);
                if us < best_cfg.1 {
                    best_cfg = ((nsg, nr0), us);
                }
                row.push_str(&format!("  {nsg}x{nr0}={us:6.1}({:.2}x)", leg_us / us));
            }
            dsl::set_pq2_nsg(None);
            dsl::set_pq2_nr0(None);
            row.push_str(&format!(
                "  | best={}x{} ({:.2}x)",
                best_cfg.0 .0,
                best_cfg.0 .1,
                leg_us / best_cfg.1
            ));
            eprintln!("{row}");
        }
    }
    dsl::set_ptq1_impl(QkImpl::Legacy);
}

#[test]
#[ignore]
fn ptq1_dsl_vs_legacy_gemm_bench() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    const ITERS: usize = 8;
    let dev = Device::new_metal(0).unwrap();
    let cases = [
        (512usize, 5120usize, 5120usize, DType::F32),
        (512, 5120, 5120, DType::BF16),
        (512, 17408, 5120, DType::F32),
        (512, 17408, 5120, DType::BF16),
        (512, 5120, 17408, DType::F32),
        (2048, 5120, 5120, DType::BF16),
        (2048, 17408, 5120, DType::BF16),
    ];
    eprintln!("=== PTQ1_0 Prefill GEMM A/B ===");
    for (m, n, k, dt) in cases {
        let mm = matmul(&dev, &bench_raw_ptq1(n, k, (m + n + k) as u64), n, k);
        let x = Tensor::from_vec(act(m * k), (m, k), &dev)
            .unwrap()
            .to_dtype(dt)
            .unwrap();
        let mut res = Vec::new();
        for imp in [QkImpl::Legacy, QkImpl::Dsl, QkImpl::Legacy, QkImpl::Dsl] {
            dsl::set_ptq1_impl(imp);
            let _ = mm.forward(&x).unwrap();
            dev.synchronize().unwrap();
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let t = std::time::Instant::now();
                for _ in 0..ITERS {
                    let _ = mm.forward(&x).unwrap();
                }
                dev.synchronize().unwrap();
                best = best.min(t.elapsed().as_secs_f64() * 1e6 / ITERS as f64);
            }
            res.push(best);
        }
        let leg = res[0].min(res[2]);
        let dsl_t = res[1].min(res[3]);
        let tflops = (2.0 * m as f64 * n as f64 * k as f64) / (dsl_t * 1e6);
        eprintln!(
            "m={m:<4} {n:>5}x{k:<5} {dt:?}  leg={leg:8.1}us  dsl={dsl_t:8.1}us  ratio={:.3}x  dsl_tflops={tflops:.2}",
            leg / dsl_t
        );
    }
    dsl::set_ptq1_impl(QkImpl::Legacy);
}
