//! Multi-bit (INT2/INT4/INT8) grouped symmetric quantization of weight deltas.
//!
//! Mirrors `~/work/hanzo/engine/hanzo-quant/src/deltaquant.rs` but uses the
//! candle scale convention and integrates with [`crate::storage`] for disk
//! IO. Group size defaults to 128, matching candle's k-quant group convention.
//!
//! Symmetric per-group quant:
//!
//! ```text
//! qmax     = 2^(bits-1) - 1
//! scale[g] = max(|x[g]|) / qmax
//! q[i]     = round(x[i] / scale[g]) clamped to [qmin, qmax]
//! ```

use candle_core::{quantized::QTensor, DType, Device, Tensor};
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// Supported bit widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantBits {
    Int2,
    Int4,
    Int8,
}

impl QuantBits {
    pub fn n(&self) -> u8 {
        match self {
            QuantBits::Int2 => 2,
            QuantBits::Int4 => 4,
            QuantBits::Int8 => 8,
        }
    }
    pub fn qmax(&self) -> i32 {
        match self {
            QuantBits::Int2 => 1,
            QuantBits::Int4 => 7,
            QuantBits::Int8 => 127,
        }
    }
    pub fn qmin(&self) -> i32 {
        match self {
            QuantBits::Int2 => -2,
            QuantBits::Int4 => -8,
            QuantBits::Int8 => -128,
        }
    }
    pub fn try_from_u8(b: u8) -> Result<Self> {
        match b {
            2 => Ok(QuantBits::Int2),
            4 => Ok(QuantBits::Int4),
            8 => Ok(QuantBits::Int8),
            _ => Err(Error::InvalidBits(b)),
        }
    }
}

/// Header for a DeltaQuant adapter — serialized via bincode in
/// [`crate::storage::write_adapter`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaQuantHeader {
    pub bits: u8, // 2 / 4 / 8
    pub group_size: u32,
    pub scales: Vec<f32>,
    pub shape: Vec<usize>,
    pub numel: usize,
}

/// A grouped quantized delta.
#[derive(Debug, Clone)]
pub struct DeltaQuantAdapter {
    pub header: DeltaQuantHeader,
    /// Packed quantized values.
    pub packed: Vec<u8>,
}

impl DeltaQuantAdapter {
    pub fn compress_against_full(
        full: &Tensor,
        base: &Tensor,
        bits: QuantBits,
        group_size: Option<usize>,
    ) -> Result<Self> {
        let fs = full.dims().to_vec();
        let bs = base.dims().to_vec();
        if fs != bs {
            return Err(Error::ShapeMismatch { full: fs, base: bs });
        }
        let dev = full.device();
        let full_f32 = full.to_dtype(DType::F32)?.to_device(dev)?;
        let base_f32 = base.to_dtype(DType::F32)?.to_device(dev)?;
        let delta = (&full_f32 - &base_f32)?;
        Self::compress_delta(&delta, bits, group_size)
    }

    pub fn compress(full: &Tensor, base: &QTensor, bits: QuantBits, gs: Option<usize>) -> Result<Self> {
        let base_t = base.dequantize(&base.device())?;
        Self::compress_against_full(full, &base_t, bits, gs)
    }

    pub fn compress_delta(delta: &Tensor, bits: QuantBits, group_size: Option<usize>) -> Result<Self> {
        let shape = delta.dims().to_vec();
        let flat: Vec<f32> = delta.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
        let numel = flat.len();
        if numel == 0 {
            return Err(Error::Empty("DeltaQuantAdapter: zero elements"));
        }
        let gs = group_size.unwrap_or(128).max(1);
        let num_groups = numel.div_ceil(gs);
        let qmax = bits.qmax() as f32;
        let qmin_i = bits.qmin();
        let qmax_i = bits.qmax();

        let mut scales = Vec::with_capacity(num_groups);
        let mut quantized: Vec<i8> = Vec::with_capacity(numel);

        for g in 0..num_groups {
            let start = g * gs;
            let end = (start + gs).min(numel);
            let group = &flat[start..end];
            let abs_max = group.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
            let scale = if abs_max < 1e-12 { 1e-8 } else { abs_max / qmax };
            scales.push(scale);
            for &x in group {
                let q = (x / scale).round() as i32;
                quantized.push(q.clamp(qmin_i, qmax_i) as i8);
            }
        }

        let packed = pack(bits, &quantized);
        Ok(Self {
            header: DeltaQuantHeader {
                bits: bits.n(),
                group_size: gs as u32,
                scales,
                shape,
                numel,
            },
            packed,
        })
    }

    pub fn decode(&self, device: &Device) -> Result<Tensor> {
        let bits = QuantBits::try_from_u8(self.header.bits)?;
        let unpacked = unpack(bits, &self.packed, self.header.numel);
        let gs = self.header.group_size as usize;
        let mut out = Vec::with_capacity(self.header.numel);
        for i in 0..self.header.numel {
            let g = i / gs;
            out.push(unpacked[i] as f32 * self.header.scales[g]);
        }
        let t = Tensor::from_vec(out, self.header.shape.as_slice(), device)?;
        Ok(t)
    }

    pub fn apply_to(&self, base: &QTensor) -> Result<QTensor> {
        let base_t = base.dequantize(&base.device())?;
        let delta = self.decode(&base.device())?;
        let merged = (&base_t + &delta)?;
        Ok(QTensor::quantize(&merged, base.dtype())?)
    }

    pub fn compression_ratio(&self) -> f32 {
        let original_bits = self.header.numel as f32 * 32.0;
        let scale_bits = self.header.scales.len() as f32 * 32.0;
        let packed_bits = self.packed.len() as f32 * 8.0;
        let meta = (self.header.shape.len() as f32 * 32.0) + 64.0 + 16.0;
        original_bits / (scale_bits + packed_bits + meta)
    }
}

fn pack(bits: QuantBits, values: &[i8]) -> Vec<u8> {
    match bits {
        QuantBits::Int8 => values.iter().map(|&x| x as u8).collect(),
        QuantBits::Int4 => {
            let n = values.len().div_ceil(2);
            let mut out = vec![0u8; n];
            for (i, &v) in values.iter().enumerate() {
                let nib = (v as u8) & 0x0F;
                if i % 2 == 0 {
                    out[i / 2] |= nib;
                } else {
                    out[i / 2] |= nib << 4;
                }
            }
            out
        }
        QuantBits::Int2 => {
            let n = values.len().div_ceil(4);
            let mut out = vec![0u8; n];
            for (i, &v) in values.iter().enumerate() {
                let two = (v as u8) & 0x03;
                let shift = (i % 4) * 2;
                out[i / 4] |= two << shift;
            }
            out
        }
    }
}

fn unpack(bits: QuantBits, packed: &[u8], numel: usize) -> Vec<i8> {
    match bits {
        QuantBits::Int8 => packed.iter().take(numel).map(|&b| b as i8).collect(),
        QuantBits::Int4 => {
            let mut out = Vec::with_capacity(numel);
            for i in 0..numel {
                let byte = packed[i / 2];
                let nib = if i % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
                let signed = if nib & 0x08 != 0 { (nib | 0xF0) as i8 } else { nib as i8 };
                out.push(signed);
            }
            out
        }
        QuantBits::Int2 => {
            let mut out = Vec::with_capacity(numel);
            for i in 0..numel {
                let byte = packed[i / 4];
                let shift = (i % 4) * 2;
                let two = (byte >> shift) & 0x03;
                let signed = if two & 0x02 != 0 { (two | 0xFC) as i8 } else { two as i8 };
                out.push(signed);
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn int8_round_trip_close_to_lossless() {
        let dev = Device::Cpu;
        let v: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.001).collect();
        let t = Tensor::from_vec(v.clone(), 256, &dev).unwrap();
        let zero = Tensor::zeros(256, DType::F32, &dev).unwrap();
        let dq = DeltaQuantAdapter::compress_against_full(&t, &zero, QuantBits::Int8, Some(128))
            .unwrap();
        let back: Vec<f32> = dq.decode(&dev).unwrap().to_vec1().unwrap();
        for (a, b) in v.iter().zip(back.iter()) {
            assert!((a - b).abs() < 2e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn int4_round_trip_bounded_by_half_scale() {
        let dev = Device::Cpu;
        let v: Vec<f32> = (0..128).map(|i| ((i % 15) as f32 - 7.0) * 0.1).collect();
        let t = Tensor::from_vec(v.clone(), 128, &dev).unwrap();
        let zero = Tensor::zeros(128, DType::F32, &dev).unwrap();
        let dq = DeltaQuantAdapter::compress_against_full(&t, &zero, QuantBits::Int4, Some(128))
            .unwrap();
        let back: Vec<f32> = dq.decode(&dev).unwrap().to_vec1().unwrap();
        let max_scale = dq.header.scales.iter().cloned().fold(0.0_f32, f32::max);
        for (a, b) in v.iter().zip(back.iter()) {
            let err = (a - b).abs();
            assert!(err <= 0.5 * max_scale + 1e-6, "err {err} > tol");
        }
    }
}
