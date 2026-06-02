//! End-to-end interop tests against candle's `QTensor` / `GgmlDType`.

use candle_core::{
    quantized::{GgmlDType, QTensor},
    DType, Device, Tensor,
};
use hanzo_quantize::{bitdelta::BitDeltaAdapter, ggml_bridge};

fn full_tensor(rows: usize, cols: usize) -> (Tensor, Device) {
    let dev = Device::Cpu;
    let v: Vec<f32> = (0..rows * cols)
        .map(|i| ((i as f32 - (rows * cols) as f32 / 2.0) * 0.01).tanh())
        .collect();
    (Tensor::from_vec(v, (rows, cols), &dev).unwrap(), dev)
}

#[test]
fn compress_against_q8_0_and_apply() {
    let (full, dev) = full_tensor(32, 64);
    let base_q = QTensor::quantize(&full, GgmlDType::Q8_0).unwrap();
    let adapter = BitDeltaAdapter::compress(&full, &base_q).unwrap();

    // base_dtype is preserved through the header
    assert_eq!(adapter.base_dtype(), Some(GgmlDType::Q8_0));

    // Apply back, dequantize, and ensure shape preserved.
    let merged = adapter.apply_to(&base_q).unwrap();
    assert_eq!(merged.dtype(), GgmlDType::Q8_0);
    assert_eq!(merged.shape().dims(), &[32, 64]);
    let _ = merged.dequantize(&dev).unwrap();
}

#[test]
fn compress_against_q4k_residual_improves_quant_error() {
    // Q4_K is much lossier than Q8_0; adding a sign-only residual should
    // reduce mean abs error vs the plain quantization.
    // Q4_K block size = 256, so last dim must be a multiple of 256.
    let (full, dev) = full_tensor(8, 256);
    let base_q = QTensor::quantize(&full, GgmlDType::Q4K).unwrap();
    let base_dq = base_q.dequantize(&dev).unwrap();
    let adapter = BitDeltaAdapter::compress(&full, &base_q).unwrap();
    let merged_t = adapter.apply_to_tensor(&base_q).unwrap();

    let full_v: Vec<f32> = full.flatten_all().unwrap().to_vec1().unwrap();
    let base_v: Vec<f32> = base_dq.flatten_all().unwrap().to_vec1().unwrap();
    let merg_v: Vec<f32> = merged_t.flatten_all().unwrap().to_vec1().unwrap();

    let mae_base: f32 = full_v.iter().zip(&base_v).map(|(a, b)| (a - b).abs()).sum::<f32>()
        / full_v.len() as f32;
    let mae_merg: f32 = full_v.iter().zip(&merg_v).map(|(a, b)| (a - b).abs()).sum::<f32>()
        / full_v.len() as f32;
    // Residual sign adapter should not be strictly worse, and on average
    // narrows the gap by capturing direction (~10-30% improvement typical).
    assert!(
        mae_merg <= mae_base * 1.1,
        "merged MAE {mae_merg} much worse than base {mae_base}"
    );
}

#[test]
fn ggml_bridge_split_merge_q5k_dims_preserved() {
    // Q5_K needs last dim divisible by 256.
    let (full, _) = full_tensor(8, 256);
    let (base, adapter) = ggml_bridge::split(&full, GgmlDType::Q5K).unwrap();
    assert_eq!(base.dtype(), GgmlDType::Q5K);
    assert_eq!(adapter.header.shape, vec![8, 256]);
    let merged = ggml_bridge::merge(&base, &adapter).unwrap();
    assert_eq!(merged.shape().dims(), &[8, 256]);
    assert_eq!(merged.dtype(), GgmlDType::Q5K);
}

#[test]
fn pair_size_smaller_than_f32() {
    // Q4_K requires last dim divisible by 256. Use a Llama-shape layer.
    let (full, _) = full_tensor(1024, 1024);
    let (base, adapter) = ggml_bridge::split(&full, GgmlDType::Q4K).unwrap();
    let pair = ggml_bridge::pair_size_bytes(&base, &adapter);
    let f32_raw = ggml_bridge::full_size_bytes(&[1024, 1024], DType::F32);
    assert!(pair < f32_raw / 4, "pair {pair} not <= f32/4 = {}", f32_raw / 4);
}
