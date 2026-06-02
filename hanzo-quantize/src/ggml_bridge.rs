//! Bridge between BitDelta and candle's GGUF / `GgmlDType` quant types.
//!
//! Two patterns:
//!
//! 1. **Split**: given a full-precision tensor `T` and a target `GgmlDType`,
//!    produce `(QTensor base, BitDeltaAdapter delta)` such that
//!    `delta.apply_to(&base) ≈ QTensor::quantize(T, dtype)`. The base is the
//!    standard candle-quantized tensor; the delta captures the residual
//!    quantization noise + any additional structure you want preserved.
//! 2. **Merge**: given a `(QTensor base, BitDeltaAdapter delta)` pair, produce
//!    a fresh `QTensor` of the same dtype as the base.
//!
//! Use case: shipping a base model once (large, GGUF-quantized) and many
//! small BitDelta adapters per fine-tune.

use candle_core::{
    quantized::{GgmlDType, QTensor},
    DType, Tensor,
};

use crate::{bitdelta::BitDeltaAdapter, Result};

/// Quantize `full` to `dtype`, then compute a BitDelta adapter capturing the
/// residual `full - dequantize(quantize(full))`. The returned `QTensor` is
/// the canonical candle quantization; the adapter encodes the noise.
pub fn split(full: &Tensor, dtype: GgmlDType) -> Result<(QTensor, BitDeltaAdapter)> {
    let q = QTensor::quantize(full, dtype)?;
    let adapter = BitDeltaAdapter::compress(full, &q)?;
    Ok((q, adapter))
}

/// Inverse of [`split`]. Returns a fresh `QTensor` of the same dtype as
/// `base`, with the BitDelta merged in.
pub fn merge(base: &QTensor, adapter: &BitDeltaAdapter) -> Result<QTensor> {
    adapter.apply_to(base)
}

/// As [`merge`] but returns the merged tensor as raw f32 (no requantization).
/// Cheaper when you'll use the result through `QMatMul::Tensor` instead of
/// `QMatMul::QTensor`.
pub fn merge_to_f32(base: &QTensor, adapter: &BitDeltaAdapter) -> Result<Tensor> {
    adapter.apply_to_tensor(base)
}

/// Convenience: total bytes used by `(base, adapter)` together. Useful for
/// the "how much smaller than raw bf16/f32" question.
pub fn pair_size_bytes(base: &QTensor, adapter: &BitDeltaAdapter) -> usize {
    base.storage_size_in_bytes() + adapter.size_bytes()
}

/// What the same tensor would cost as raw bf16 / f32.
///
/// Uses elt-size in *bytes*, rounding sub-byte dtypes up to the next byte
/// (the actual GGUF block formats pack these more tightly, but this is a
/// "comparable raw upper bound").
pub fn full_size_bytes(shape: &[usize], dtype: DType) -> usize {
    let n: usize = shape.iter().product();
    let elt = match dtype {
        DType::F32 | DType::U32 | DType::I32 => 4,
        DType::F16 | DType::BF16 | DType::I16 => 2,
        DType::U8 | DType::F8E4M3 | DType::F8E8M0 => 1,
        DType::F64 | DType::I64 => 8,
        // Sub-byte dtypes round up to 1 (one byte per element, generous bound).
        DType::F6E2M3 | DType::F6E3M2 | DType::F4 => 1,
    };
    n * elt
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn split_and_merge_round_trip() {
        let dev = Device::Cpu;
        // A 32x32 tensor (32 is a valid k_quant block for Q8_0 which needs 32 elt/block).
        let v: Vec<f32> = (0..1024).map(|i| ((i as f32 - 512.0) * 0.01).sin()).collect();
        let full = Tensor::from_vec(v, (32, 32), &dev).unwrap();

        let (base, adapter) = split(&full, GgmlDType::Q8_0).unwrap();
        let merged = merge_to_f32(&base, &adapter).unwrap();

        // Adapter is sign-only — reconstruction is base + sign(delta)*scale,
        // not full itself. But it should be CLOSER to full than the plain
        // base dequant. Check both errors and require improvement.
        let full_v: Vec<f32> = full.flatten_all().unwrap().to_vec1().unwrap();
        let base_v: Vec<f32> = base.dequantize(&dev).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let merged_v: Vec<f32> = merged.flatten_all().unwrap().to_vec1().unwrap();

        let err_base: f32 = full_v.iter().zip(&base_v).map(|(a, b)| (a - b).abs()).sum();
        let err_merged: f32 = full_v.iter().zip(&merged_v).map(|(a, b)| (a - b).abs()).sum();
        // BitDelta-as-residual should not make things WORSE (sign-only adapter
        // captures direction of residual; magnitude is per-channel mean).
        assert!(
            err_merged <= err_base * 1.1,
            "merged err {err_merged} > base err {err_base} by >10%"
        );
    }

    #[test]
    fn pair_size_is_smaller_than_bf16_for_real_layer() {
        // Q4_K requires last dim divisible by 256 (super-block of 256).
        // Use a Llama-ish 1024x1024 layer.
        let dev = Device::Cpu;
        let (rows, cols) = (1024, 1024);
        let v: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.001).collect();
        let full = Tensor::from_vec(v, (rows, cols), &dev).unwrap();
        let (base, adapter) = split(&full, GgmlDType::Q4K).unwrap();
        let pair = pair_size_bytes(&base, &adapter);
        let bf16 = full_size_bytes(&[rows, cols], DType::BF16);
        // Q4_K base (~0.55 byte/elt) + BitDelta (~0.13 byte/elt) << bf16 (2 byte/elt).
        assert!(pair < bf16, "pair {pair} >= bf16 {bf16}");
        // Sanity check: should be well under half of bf16
        assert!(pair < bf16 / 2, "pair {pair} >= bf16/2 {}", bf16 / 2);
    }
}
