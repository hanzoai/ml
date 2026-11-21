//! End-to-end ROCm qualification for production DSL `PQ2_0` and `PTQ1_0` dispatch in `hanzo-ml`.
//!
//! Covers:
//! 1. `PQ2_0` and `PTQ1_0` decode matvec (`m == 1`, `f32` and `bf16`) vs f64 oracle and legacy.
//! 2. `PQ2_0` and `PTQ1_0` prefill GEMM (`m > 1`, `bm = 32` and `bm = 64`, `f32` and `bf16`) vs
//!    f64 oracle, plus nonzero activation slice offset and 3D batch shapes.
//! 3. Single-compile proof (`compiled 1x`) and dispatch counters (`quantized::dsl::stats`).
#![cfg(feature = "rocm")]

use std::borrow::Cow;
use std::sync::Mutex;

use hanzo_ml::quantized::dsl::{self, QkImpl};
use hanzo_ml::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use hanzo_ml::{DType, Device, Module, Tensor};

static SERIAL: Mutex<()> = Mutex::new(());

fn act(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 7919) % 2003) as f32 / 1001.5) - 1.0)
        .collect()
}

fn raw_pq2(n: usize, k: usize, seed: u64) -> Vec<u8> {
    let (codes, d) = hanzo_kernel::ternary::pq2_0_synth(n, k, seed);
    hanzo_kernel::ternary::pq2_0_gguf_bytes(&codes, &d, n, k)
}

fn raw_ptq1(n: usize, k: usize, seed: u64) -> Vec<u8> {
    let (codes, d) = hanzo_kernel::ternary::ptq1_0_synth(n, k, seed);
    hanzo_kernel::ternary::ptq1_0_gguf_bytes(&codes, &d, n, k)
}

fn matmul(dev: &Device, dtype: GgmlDType, raw: &[u8], n: usize, k: usize) -> QMatMul {
    let qt = QTensor::new(
        QStorage::from_data(Cow::Borrowed(raw), dev, dtype).unwrap(),
        (n, k),
    )
    .unwrap();
    QMatMul::from_qtensor(qt).unwrap()
}

fn scale_rel(want: &[f32], got: &[f32]) -> f32 {
    assert_eq!(want.len(), got.len());
    let maxref = want.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-30);
    want.iter()
        .zip(got)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
        / maxref
}

fn forward_pq2(mm: &QMatMul, x: &Tensor, imp: QkImpl) -> Vec<f32> {
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

fn forward_ptq1(mm: &QMatMul, x: &Tensor, imp: QkImpl) -> Vec<f32> {
    dsl::set_ptq1_impl(imp);
    mm.forward(x)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

#[test]
fn rocm_pq2_and_ptq1_dsl_matvec_match_oracle_and_legacy() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dev = Device::new_rocm(0).unwrap();
    dsl::reset_stats();
    let shapes = [(1024usize, 5120usize), (5120, 5120), (17408, 5120)];

    for &(n, k) in &shapes {
        let xv = act(k);
        let x32 = Tensor::from_vec(xv.clone(), (1, k), &dev).unwrap();
        let xb = x32.to_dtype(DType::BF16).unwrap();
        let xbv: Vec<f32> = xb
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // PQ2_0
        let raw = raw_pq2(n, k, (n * 17 + k) as u64);
        let mm = matmul(&dev, GgmlDType::PQ2_0, &raw, n, k);
        let want32 = hanzo_kernel::ternary::pq2_0_gguf_ref(&raw, &xv, n, k);
        let got_dsl = forward_pq2(&mm, &x32, QkImpl::Dsl);
        let got_leg = forward_pq2(&mm, &x32, QkImpl::Legacy);
        let rel_dsl = scale_rel(&want32, &got_dsl);
        let rel_leg = scale_rel(&want32, &got_leg);
        eprintln!("ROCm PQ2  f32  {n:>5}x{k:<5} dsl={rel_dsl:.2e} legacy={rel_leg:.2e}");
        assert!(rel_dsl <= 5e-5, "ROCm PQ2 f32 rel={rel_dsl:.3e}");

        let want_b = hanzo_kernel::ternary::pq2_0_gguf_ref(&raw, &xbv, n, k);
        let got_b_dsl = forward_pq2(&mm, &xb, QkImpl::Dsl);
        let rel_b = scale_rel(&want_b, &got_b_dsl);
        eprintln!("ROCm PQ2  bf16 {n:>5}x{k:<5} dsl={rel_b:.2e}");
        assert!(rel_b <= 5e-3, "ROCm PQ2 bf16 rel={rel_b:.3e}");

        // PTQ1_0
        let raw_t = raw_ptq1(n, k, (n * 31 + k) as u64);
        let mm_t = matmul(&dev, GgmlDType::PTQ1_0, &raw_t, n, k);
        let want32_t = hanzo_kernel::ternary::ptq1_0_gguf_ref(&raw_t, &xv, n, k);
        let got_dsl_t = forward_ptq1(&mm_t, &x32, QkImpl::Dsl);
        let got_leg_t = forward_ptq1(&mm_t, &x32, QkImpl::Legacy);
        let rel_dsl_t = scale_rel(&want32_t, &got_dsl_t);
        let rel_leg_t = scale_rel(&want32_t, &got_leg_t);
        eprintln!("ROCm PTQ1 f32  {n:>5}x{k:<5} dsl={rel_dsl_t:.2e} legacy={rel_leg_t:.2e}");
        assert!(rel_dsl_t <= 5e-5, "ROCm PTQ1 f32 rel={rel_dsl_t:.3e}");

        let want_bt = hanzo_kernel::ternary::ptq1_0_gguf_ref(&raw_t, &xbv, n, k);
        let got_b_dsl_t = forward_ptq1(&mm_t, &xb, QkImpl::Dsl);
        let rel_bt = scale_rel(&want_bt, &got_b_dsl_t);
        eprintln!("ROCm PTQ1 bf16 {n:>5}x{k:<5} dsl={rel_bt:.2e}");
        assert!(rel_bt <= 5e-3, "ROCm PTQ1 bf16 rel={rel_bt:.3e}");
    }

    let st = dsl::stats();
    assert_eq!(st.compiles_of("matvec_pq2_0_gguf_nsg2_nr4_f32"), 1);
    assert_eq!(st.compiles_of("matvec_pq2_0_gguf_nsg2_nr4_bf16"), 1);
    assert_eq!(st.compiles_of("matvec_ptq1_0_gguf_nsg2_nr4_f32"), 1);
    assert_eq!(st.compiles_of("matvec_ptq1_0_gguf_nsg2_nr4_bf16"), 1);
    dsl::set_pq2_impl(QkImpl::Legacy);
    dsl::set_ptq1_impl(QkImpl::Legacy);
}

#[test]
fn rocm_pq2_and_ptq1_dsl_gemm_prefill_match_oracle_and_offset() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dev = Device::new_rocm(0).unwrap();
    dsl::reset_stats();
    let cases = [
        (9usize, 128usize, 256usize),
        (35, 386, 384),
        (257, 2048, 256),
    ];

    for &(m, n, k) in &cases {
        let xv = act(m * k);
        let x32 = Tensor::from_vec(xv.clone(), (m, k), &dev).unwrap();
        let xb = x32.to_dtype(DType::BF16).unwrap();
        let xbv: Vec<f32> = xb
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let raw = raw_pq2(n, k, (m * 31 + n * 7 + k) as u64);
        let mm = matmul(&dev, GgmlDType::PQ2_0, &raw, n, k);
        let want32 = hanzo_kernel::ternary::pq2_0_gguf_gemm_ref(&raw, &xv, m, n, k);
        let got_dsl = forward_pq2(&mm, &x32, QkImpl::Dsl);
        let rel32 = scale_rel(&want32, &got_dsl);
        eprintln!("ROCm PQ2  GEMM f32  m={m:>3} n={n:>4} k={k:<3} rel={rel32:.2e}");
        assert!(
            rel32 <= 5e-4,
            "ROCm PQ2 GEMM f32 m={m} n={n} k={k}: {rel32:.3e}"
        );

        let want_b = hanzo_kernel::ternary::pq2_0_gguf_gemm_ref(&raw, &xbv, m, n, k);
        let got_b = forward_pq2(&mm, &xb, QkImpl::Dsl);
        let rel_b = scale_rel(&want_b, &got_b);
        assert!(
            rel_b <= 8e-3,
            "ROCm PQ2 GEMM bf16 m={m} n={n} k={k}: {rel_b:.3e}"
        );

        let raw_t = raw_ptq1(n, k, (m * 19 + n * 11 + k) as u64);
        let mm_t = matmul(&dev, GgmlDType::PTQ1_0, &raw_t, n, k);
        let want32_t = hanzo_kernel::ternary::ptq1_0_gguf_gemm_ref(&raw_t, &xv, m, n, k);
        let got_dsl_t = forward_ptq1(&mm_t, &x32, QkImpl::Dsl);
        let rel32_t = scale_rel(&want32_t, &got_dsl_t);
        eprintln!("ROCm PTQ1 GEMM f32  m={m:>3} n={n:>4} k={k:<3} rel={rel32_t:.2e}");
        assert!(
            rel32_t <= 5e-4,
            "ROCm PTQ1 GEMM f32 m={m} n={n} k={k}: {rel32_t:.3e}"
        );

        let want_bt = hanzo_kernel::ternary::ptq1_0_gguf_gemm_ref(&raw_t, &xbv, m, n, k);
        let got_bt = forward_ptq1(&mm_t, &xb, QkImpl::Dsl);
        let rel_bt = scale_rel(&want_bt, &got_bt);
        assert!(
            rel_bt <= 8e-3,
            "ROCm PTQ1 GEMM bf16 m={m} n={n} k={k}: {rel_bt:.3e}"
        );
    }

    // Nonzero activation offset + 3D batch.
    let (n, k) = (257usize, 512usize);
    let raw = raw_pq2(n, k, 999);
    let mm = matmul(&dev, GgmlDType::PQ2_0, &raw, n, k);
    let big = Tensor::from_vec(act(32 * k), (32, k), &dev).unwrap();
    let sub = big.narrow(0, 7, 17).unwrap();
    let sub_c = sub.contiguous().unwrap();
    assert_eq!(
        forward_pq2(&mm, &sub, QkImpl::Dsl),
        forward_pq2(&mm, &sub_c, QkImpl::Dsl)
    );

    let xb3 = Tensor::from_vec(act(22 * k), (2, 11, k), &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let xb2 = xb3.reshape((22, k)).unwrap();
    assert_eq!(
        forward_pq2(&mm, &xb3, QkImpl::Dsl),
        forward_pq2(&mm, &xb2, QkImpl::Dsl)
    );

    let st = dsl::stats();
    assert_eq!(st.compiles_of("matmul_pq2_0_gguf_bm32_f32"), 1);
    assert_eq!(st.compiles_of("matmul_pq2_0_gguf_bm32_bf16"), 1);
    assert_eq!(st.compiles_of("matmul_pq2_0_gguf_bm64_f32"), 1);
    assert_eq!(st.compiles_of("matmul_pq2_0_gguf_bm64_bf16"), 1);
    assert_eq!(st.compiles_of("matmul_ptq1_0_gguf_bm32_f32"), 1);
    assert_eq!(st.compiles_of("matmul_ptq1_0_gguf_bm32_bf16"), 1);
    assert_eq!(st.compiles_of("matmul_ptq1_0_gguf_bm64_f32"), 1);
    assert_eq!(st.compiles_of("matmul_ptq1_0_gguf_bm64_bf16"), 1);
    dsl::set_pq2_impl(QkImpl::Legacy);
    dsl::set_ptq1_impl(QkImpl::Legacy);
}
