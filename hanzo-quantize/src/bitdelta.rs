//! Per-channel BitDelta, native-candle.
//!
//! Algorithm:
//!
//! 1. `delta = full - base`
//! 2. For rank >= 2 tensors with shape `[out, in, ...]`, compute one scale per
//!    output channel: `scales[c] = mean(|delta[c, ..]|)`. For rank < 2 fall
//!    back to a single per-tensor scale (mirrors the simpler reference impl
//!    in `~/work/hanzo/engine/hanzo-quant/src/bitdelta.rs`).
//! 3. `sign_bit[i] = 1 if delta[i] >= 0 else 0`, packed **MSB-first** to match
//!    the GGUF byte-order convention (bit 7 = first element of the byte, bit
//!    0 = eighth). This is the opposite of the engine's LE packing — see the
//!    "Key implementation notes" block in the task brief.
//! 4. To reconstruct: `delta_hat[c, i, ..] = (sign ? +1 : -1) * scales[c]`.
//!
//! ## Integration with `candle_core::quantized`
//!
//! - [`BitDeltaAdapter::compress`] takes a `Tensor` (the full weights) and a
//!   `&QTensor` (the candle-quantized base), dequantizes the base, computes
//!   the per-channel residual, and packs it.
//! - [`BitDeltaAdapter::apply_to`] does the inverse: dequantize the base into
//!   a `Tensor`, add the unpacked delta, and re-quantize using the **same**
//!   `GgmlDType` as the base. Caller gets a fresh `QTensor`.
//!
//! For inference that doesn't want to round-trip through requantization, use
//! [`BitDeltaAdapter::apply_to_tensor`] which returns the merged f32 tensor
//! directly (skip the requantize step — useful when you'll multiply through
//! `QMatMul::Tensor` instead of `QMatMul::QTensor`).

use candle_core::{
    quantized::{GgmlDType, QTensor},
    DType, Device, Shape, Tensor,
};
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// Header metadata that goes on disk alongside the packed sign bits. Kept
/// small so it round-trips through bincode cheaply.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitDeltaHeader {
    /// One f32 scale per output channel. For rank < 2 tensors, length 1.
    pub scales: Vec<f32>,
    /// Total number of elements (sign bits are bit-packed so we need this).
    pub numel: usize,
    /// Original tensor shape.
    pub shape: Vec<usize>,
    /// GgmlDType discriminant of the base this adapter was computed against.
    /// Stored so on-disk adapters tell you what base they belong to.
    /// 255 == "unspecified" (e.g. compressed against a raw `Tensor`).
    pub base_dtype: u32,
}

/// A BitDelta adapter: header + packed sign bits. ~32× smaller than the
/// original full-precision delta (1 bit per element + per-channel scales).
#[derive(Debug, Clone)]
pub struct BitDeltaAdapter {
    pub header: BitDeltaHeader,
    /// Packed sign bits, MSB-first. `(sign_bits[i/8] >> (7 - i%8)) & 1`.
    pub sign_bits: Vec<u8>,
}

impl BitDeltaAdapter {
    /// Compress `full - dequantize(base)` into a BitDelta adapter.
    ///
    /// `full` is expected to be the high-precision fine-tuned weight on any
    /// device; it's cast to f32 internally. `base` is a candle-quantized
    /// tensor (`QTensor::quantize(&t, dtype)`).
    pub fn compress(full: &Tensor, base: &QTensor) -> Result<Self> {
        let base_dtype = base.dtype();
        let base_dev = base.device();
        let base_t = base.dequantize(&base_dev)?;
        Self::compress_against_tensor(full, &base_t, Some(ggml_dtype_to_u32(base_dtype)))
    }

    /// Compress against a raw f32 tensor (no base quantization). Sets the
    /// header's `base_dtype` to the sentinel `u32::MAX` so consumers know
    /// there's no GgmlDType reference.
    pub fn compress_against_full(full: &Tensor, base: &Tensor) -> Result<Self> {
        Self::compress_against_tensor(full, base, None)
    }

    fn compress_against_tensor(
        full: &Tensor,
        base: &Tensor,
        base_dtype_u32: Option<u32>,
    ) -> Result<Self> {
        let fs = full.dims().to_vec();
        let bs = base.dims().to_vec();
        if fs != bs {
            return Err(Error::ShapeMismatch { full: fs, base: bs });
        }
        if fs.is_empty() {
            return Err(Error::Empty("BitDeltaAdapter::compress: scalar tensor"));
        }
        let dev = full.device();
        let full_f32 = full.to_dtype(DType::F32)?.to_device(dev)?;
        let base_f32 = base.to_dtype(DType::F32)?.to_device(dev)?;
        let delta = (&full_f32 - &base_f32)?;

        // For rank >= 2 we compute per-output-channel scales. Convention: dim 0
        // is the output dim (matches candle's `[n, k]` weight layout for linear
        // layers). For rank 1 we fall back to per-tensor.
        let shape = delta.dims().to_vec();
        let numel: usize = shape.iter().product();
        let flat: Vec<f32> = delta.flatten_all()?.to_vec1()?;

        let scales = if shape.len() >= 2 {
            let out_dim = shape[0];
            let chan_size = numel / out_dim;
            let mut s = Vec::with_capacity(out_dim);
            for c in 0..out_dim {
                let start = c * chan_size;
                let end = start + chan_size;
                let abs_sum: f32 = flat[start..end].iter().map(|x| x.abs()).sum();
                let m = (abs_sum / chan_size as f32).max(1e-8);
                s.push(m);
            }
            s
        } else {
            let abs_sum: f32 = flat.iter().map(|x| x.abs()).sum();
            vec![(abs_sum / numel as f32).max(1e-8)]
        };

        // Pack sign bits MSB-first.
        let nbytes = numel.div_ceil(8);
        let mut sign_bits = vec![0u8; nbytes];
        for (i, &v) in flat.iter().enumerate() {
            if v >= 0.0 {
                sign_bits[i / 8] |= 1u8 << (7 - i % 8);
            }
        }

        let header = BitDeltaHeader {
            scales,
            numel,
            shape,
            base_dtype: base_dtype_u32.unwrap_or(u32::MAX),
        };
        Ok(Self { header, sign_bits })
    }

    /// Decode the delta back to a full f32 tensor on `device`.
    pub fn decode(&self, device: &Device) -> Result<Tensor> {
        let shape = &self.header.shape;
        let numel = self.header.numel;
        let mut out = Vec::with_capacity(numel);

        if shape.len() >= 2 {
            let out_dim = shape[0];
            let chan_size = numel / out_dim;
            for c in 0..out_dim {
                let scale = self.header.scales[c];
                for i in 0..chan_size {
                    let idx = c * chan_size + i;
                    let bit = (self.sign_bits[idx / 8] >> (7 - idx % 8)) & 1;
                    out.push(if bit == 1 { scale } else { -scale });
                }
            }
        } else {
            let scale = self.header.scales[0];
            for i in 0..numel {
                let bit = (self.sign_bits[i / 8] >> (7 - i % 8)) & 1;
                out.push(if bit == 1 { scale } else { -scale });
            }
        }

        let t = Tensor::from_vec(out, Shape::from(shape.clone()), device)?;
        Ok(t)
    }

    /// Apply the adapter on top of a quantized base. Returns a fresh
    /// `QTensor` of the **same dtype** as the input base (requantizes the
    /// merged tensor).
    pub fn apply_to(&self, base: &QTensor) -> Result<QTensor> {
        let merged = self.apply_to_tensor(base)?;
        let q = QTensor::quantize(&merged, base.dtype())?;
        Ok(q)
    }

    /// Apply the adapter to a quantized base and return the dequantized,
    /// merged f32 tensor. Cheaper than [`apply_to`] when you don't need to
    /// requantize for storage.
    pub fn apply_to_tensor(&self, base: &QTensor) -> Result<Tensor> {
        let dev = base.device();
        let base_t = base.dequantize(&dev)?;
        if base_t.dims() != self.header.shape.as_slice() {
            return Err(Error::ShapeMismatch {
                full: self.header.shape.clone(),
                base: base_t.dims().to_vec(),
            });
        }
        let delta = self.decode(&dev)?;
        Ok((&base_t + &delta)?)
    }

    /// Apply the adapter on top of a raw f32 base tensor (no requantize).
    pub fn apply_to_full(&self, base: &Tensor) -> Result<Tensor> {
        if base.dims() != self.header.shape.as_slice() {
            return Err(Error::ShapeMismatch {
                full: self.header.shape.clone(),
                base: base.dims().to_vec(),
            });
        }
        let dev = base.device();
        let delta = self.decode(dev)?;
        Ok((&base.to_dtype(DType::F32)? + &delta)?)
    }

    /// Compression ratio against a raw f32 delta. Counts header overhead.
    pub fn compression_ratio(&self) -> f32 {
        let original_bits = self.header.numel as f32 * 32.0;
        let scale_bits = self.header.scales.len() as f32 * 32.0;
        let sign_bits = self.sign_bits.len() as f32 * 8.0;
        // Header overhead: shape (32 bits/dim) + numel (32) + base_dtype (32).
        let meta = (self.header.shape.len() as f32 * 32.0) + 64.0;
        original_bits / (scale_bits + sign_bits + meta)
    }

    /// Size in bytes of header + packed payload (the "on-disk size").
    pub fn size_bytes(&self) -> usize {
        let header_bytes = bincode::serialized_size(&self.header).unwrap_or(0) as usize;
        header_bytes + self.sign_bits.len()
    }

    /// Returns the recovered `GgmlDType` if this adapter was compressed
    /// against a `QTensor`. `None` if it was compressed against a raw tensor.
    pub fn base_dtype(&self) -> Option<GgmlDType> {
        if self.header.base_dtype == u32::MAX {
            None
        } else {
            ggml_dtype_from_u32(self.header.base_dtype)
        }
    }
}

// `GgmlDType::to_u32` / `from_u32` are `pub(crate)` in candle, so we inline
// the GGML-numbered mapping here. Matches:
// https://github.com/ggerganov/ggml/blob/29d87fc/include/ggml.h#L389
fn ggml_dtype_to_u32(d: GgmlDType) -> u32 {
    match d {
        GgmlDType::F32 => 0,
        GgmlDType::F16 => 1,
        GgmlDType::Q4_0 => 2,
        GgmlDType::Q4_1 => 3,
        GgmlDType::Q5_0 => 6,
        GgmlDType::Q5_1 => 7,
        GgmlDType::Q8_0 => 8,
        GgmlDType::Q8_1 => 9,
        GgmlDType::Q2K => 10,
        GgmlDType::Q3K => 11,
        GgmlDType::Q4K => 12,
        GgmlDType::Q5K => 13,
        GgmlDType::Q6K => 14,
        GgmlDType::Q8K => 15,
        GgmlDType::BF16 => 30,
    }
}

fn ggml_dtype_from_u32(u: u32) -> Option<GgmlDType> {
    Some(match u {
        0 => GgmlDType::F32,
        1 => GgmlDType::F16,
        2 => GgmlDType::Q4_0,
        3 => GgmlDType::Q4_1,
        6 => GgmlDType::Q5_0,
        7 => GgmlDType::Q5_1,
        8 => GgmlDType::Q8_0,
        9 => GgmlDType::Q8_1,
        10 => GgmlDType::Q2K,
        11 => GgmlDType::Q3K,
        12 => GgmlDType::Q4K,
        13 => GgmlDType::Q5K,
        14 => GgmlDType::Q6K,
        15 => GgmlDType::Q8K,
        30 => GgmlDType::BF16,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn cpu() -> Device {
        Device::Cpu
    }

    #[test]
    fn round_trip_preserves_signs() {
        let dev = cpu();
        let n = 128;
        let v: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 0.1 } else { -0.3 }).collect();
        let full = Tensor::from_vec(v.clone(), (8, 16), &dev).unwrap();
        let base = Tensor::zeros((8, 16), DType::F32, &dev).unwrap();

        let adapter = BitDeltaAdapter::compress_against_full(&full, &base).unwrap();
        let decoded = adapter.decode(&dev).unwrap();
        let out: Vec<f32> = decoded.flatten_all().unwrap().to_vec1().unwrap();

        for (i, &o) in out.iter().enumerate() {
            assert!(o.signum() == v[i].signum() || v[i] == 0.0, "sign mismatch at {i}");
        }
    }

    #[test]
    fn msb_first_sign_packing() {
        let dev = cpu();
        // 8 positives -> first byte should be 0xFF; this is the same for
        // both LE and MSB-first (all 1s). Use 9 elements with the last as
        // negative to actually distinguish: byte0 = 0xFF (all+), byte1 has
        // a 0 at bit 7 (the *first* bit of byte 1, which is MSB) -> 0x00
        // since only one element fits in byte 1. Use 16 elements with
        // distinctive pattern instead.
        // Pattern: + - + - + - + -  (byte0 should be 0b10101010 = 0xAA)
        let v: Vec<f32> =
            vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let full = Tensor::from_vec(v, 8, &dev).unwrap();
        let base = Tensor::zeros(8, DType::F32, &dev).unwrap();
        let adapter = BitDeltaAdapter::compress_against_full(&full, &base).unwrap();
        // MSB-first: bit 7 of byte 0 = element 0; bit 0 of byte 0 = element 7.
        // So bits high-to-low are 1 0 1 0 1 0 1 0 = 0b10101010 = 0xAA.
        assert_eq!(adapter.sign_bits, vec![0xAA]);
    }

    #[test]
    fn per_channel_scale_is_per_row() {
        let dev = cpu();
        // 2x4 tensor. Row 0 has |delta|=0.1, row 1 has |delta|=0.5. Expect
        // two distinct scales.
        let v: Vec<f32> = vec![
            0.1, -0.1, 0.1, -0.1, //
            0.5, -0.5, 0.5, -0.5,
        ];
        let full = Tensor::from_vec(v, (2, 4), &dev).unwrap();
        let base = Tensor::zeros((2, 4), DType::F32, &dev).unwrap();
        let adapter = BitDeltaAdapter::compress_against_full(&full, &base).unwrap();

        assert_eq!(adapter.header.scales.len(), 2);
        assert!((adapter.header.scales[0] - 0.1).abs() < 1e-6);
        assert!((adapter.header.scales[1] - 0.5).abs() < 1e-6);

        // Decode and verify magnitudes match per-row.
        let dec: Vec<f32> = adapter.decode(&dev).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        for i in 0..4 {
            assert!((dec[i].abs() - 0.1).abs() < 1e-6);
        }
        for i in 4..8 {
            assert!((dec[i].abs() - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn rank1_falls_back_to_per_tensor_scale() {
        let dev = cpu();
        let v: Vec<f32> = (0..32).map(|i| if i % 2 == 0 { 0.2 } else { -0.4 }).collect();
        let full = Tensor::from_vec(v, 32, &dev).unwrap();
        let base = Tensor::zeros(32, DType::F32, &dev).unwrap();
        let adapter = BitDeltaAdapter::compress_against_full(&full, &base).unwrap();

        assert_eq!(adapter.header.scales.len(), 1);
        // mean(|delta|) = (16*0.2 + 16*0.4)/32 = 0.3
        assert!((adapter.header.scales[0] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn compression_ratio_close_to_32_for_large_tensors() {
        let dev = cpu();
        // 64x64 = 4096 elements. With per-channel scales the ratio is bounded by
        //   raw_f32_bits / (numel + out_dim*32 + meta)
        // For 64x64: 4096*32 / (4096 + 64*32 + ~96) = 131072 / 6240 ~= 21x
        let v: Vec<f32> = (0..4096).map(|i| (i as f32 - 2048.0) * 0.001).collect();
        let full = Tensor::from_vec(v, (64, 64), &dev).unwrap();
        let base = Tensor::zeros((64, 64), DType::F32, &dev).unwrap();
        let adapter = BitDeltaAdapter::compress_against_full(&full, &base).unwrap();
        let ratio = adapter.compression_ratio();
        assert!(ratio > 18.0 && ratio < 25.0, "ratio = {ratio}");
    }

    #[test]
    fn compression_ratio_approaches_32_for_wide_tensors() {
        // 1024x4096: 1024 channels but 4M elements, so scale overhead
        // (1024*32=32768 bits) becomes negligible vs 4M sign bits.
        // Expected: 4M*32 / (4M + 32K + meta) ~= 31x.
        let dev = cpu();
        let n = 1024 * 4096;
        let v: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) * 1e-6).collect();
        let full = Tensor::from_vec(v, (1024, 4096), &dev).unwrap();
        let base = Tensor::zeros((1024, 4096), DType::F32, &dev).unwrap();
        let adapter = BitDeltaAdapter::compress_against_full(&full, &base).unwrap();
        let ratio = adapter.compression_ratio();
        assert!(ratio > 28.0 && ratio < 33.0, "ratio = {ratio}");
    }

    #[test]
    fn apply_to_full_returns_base_plus_delta() {
        let dev = cpu();
        let base = Tensor::ones((4, 4), DType::F32, &dev).unwrap();
        let full = (&base + 0.5f64).unwrap();
        let adapter = BitDeltaAdapter::compress_against_full(&full, &base).unwrap();
        let recon = adapter.apply_to_full(&base).unwrap();
        let v: Vec<f32> = recon.flatten_all().unwrap().to_vec1().unwrap();
        for x in v {
            assert!((x - 1.5).abs() < 1e-6);
        }
    }

    #[test]
    fn ggml_dtype_round_trips_through_header() {
        let dev = cpu();
        let full = Tensor::ones((32, 32), DType::F32, &dev).unwrap();
        let base_t = Tensor::zeros((32, 32), DType::F32, &dev).unwrap();
        let base_q = QTensor::quantize(&base_t, GgmlDType::Q8_0).unwrap();
        let adapter = BitDeltaAdapter::compress(&full, &base_q).unwrap();
        assert_eq!(adapter.base_dtype(), Some(GgmlDType::Q8_0));
    }
}
