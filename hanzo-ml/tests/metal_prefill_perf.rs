//! Kernel-isolated prefill/decode cost for the native Metal quantized GEMM/GEMV, on a real Apple GPU.
//!
//! Purpose (measure, don't theorize): the full-model engine prefill is far behind llama.cpp. The
//! candidate causes split cleanly in two -- (a) the tiled `kernel_mul_mm_*` simdgroup GEMM itself is
//! slow, or (b) the GEMM is fine and the deficit lives in engine orchestration (command-buffer
//! commit cadence, layout copies, activation prep, attention). This microbench isolates (a): it times
//! `QMatMul::forward` directly at production shapes, away from the model graph, so the kernel's own
//! effective throughput is visible. If GFLOP/s here is near the device roofline, the kernel is NOT
//! the bottleneck and the search moves to orchestration; if it is far below, the kernel is the lever.
//!
//! `fwd` (metal.rs) routes m==1 to the `kernel_mul_mv_*` matvec (decode) and m>1 to the
//! `kernel_mul_mm_*` simdgroup GEMM (prefill), and asserts an F32 activation -- so the activation here
//! is F32, not F16. Reports us/launch and effective GFLOP/s (2*m*n*k) per shape.
//!
//! Skips cleanly when no Metal GPU is present. Requires the `metal` feature. Not a correctness gate
//! (metal_iquant_tests covers that) -- run with `--nocapture` to read the numbers.
#![cfg(feature = "metal")]

use hanzo_ml::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use hanzo_ml::{DType, Device, Module, Tensor};
use std::time::Instant;

fn time_forward(qm: &QMatMul, x: &Tensor, dev: &Device, iters: usize) -> f64 {
    for _ in 0..3 {
        let _ = qm.forward(x).expect("forward warmup");
    }
    dev.synchronize().expect("sync");
    let t0 = Instant::now();
    for _ in 0..iters {
        let y = qm.forward(x).expect("forward");
        std::mem::drop(y);
    }
    dev.synchronize().expect("sync");
    t0.elapsed().as_secs_f64() / iters as f64 * 1e6 // us/launch
}

// Build a native Metal QTensor [n, k] of `dtype` from a random weight: quantize on CPU (the only
// place from_float lives), take the raw GGML bytes, upload straight to the Metal device -- exactly the
// GGUF model-load path.
fn metal_qmatmul(dev: &Device, n: usize, k: usize, dtype: GgmlDType) -> QMatMul {
    let wsrc = Tensor::randn(0f32, 1.0, (n, k), &Device::Cpu).expect("wsrc");
    let qt_cpu = QTensor::quantize(&wsrc, dtype).expect("cpu quantize");
    let bytes = qt_cpu.data().expect("ggml bytes"); // Cow<[u8]> over qt_cpu's GGML block bytes
    let storage = QStorage::from_data(bytes, dev, dtype).expect("upload to metal");
    let qt = QTensor::new(storage, (n, k)).expect("metal qtensor");
    QMatMul::from_qtensor(qt).expect("qmatmul")
}

fn bench_shape(dev: &Device, dtype: GgmlDType, m: usize, k: usize, n: usize, iters: usize) {
    let qm = metal_qmatmul(dev, n, k, dtype);
    // F32 activation [m, k] -- metal `fwd` asserts F32 (both the mv and mm kernels read f32 src1).
    let x = Tensor::randn(0f32, 1.0, (m, k), &Device::Cpu)
        .expect("x")
        .to_dtype(DType::F32)
        .expect("f32")
        .to_device(dev)
        .expect("to metal");
    let us = time_forward(&qm, &x, dev, iters);
    let gflops = (2.0 * m as f64 * n as f64 * k as f64) / (us * 1e-6) / 1e9;
    let path = if m == 1 { "mul_mv (decode)" } else { "mul_mm (prefill)" };
    eprintln!("{dtype:?} m={m:<4} k={k} n={n}: {us:8.1} us/launch  {gflops:8.1} GFLOP/s  [{path}]");
}

#[test]
fn metal_qmatmul_prefill_decode_cost() {
    let dev = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("no Metal GPU -- skipping");
            return;
        }
    };
    eprintln!("--- Metal QMatMul kernel-isolated cost (activation F32) ---");
    // Qwen3-4B / zen-eco-4b FFN-ish shape (K,N multiples of 256 = Q4_K/Q6_K super-block).
    for &(k, n) in &[(2560usize, 9728usize), (5120usize, 17408usize)] {
        for &dt in &[GgmlDType::Q4K, GgmlDType::Q6K] {
            bench_shape(&dev, dt, 1, k, n, 50); // decode kernel (mul_mv)
            bench_shape(&dev, dt, 512, k, n, 30); // prefill kernel (mul_mm)
        }
    }
}
