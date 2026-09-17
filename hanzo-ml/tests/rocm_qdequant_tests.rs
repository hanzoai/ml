//! Dequantizing on the device, against the CPU reference.
//!
//! A multi-row forward of a type with no native GEMM multiplies by a dense weight. That weight
//! is decoded in VRAM by `qdequant_core<WTYPE>`, which reads each element out of the same
//! per-type decode the matvec runs. Every type the unified core wires must therefore produce, on
//! the device, the very values the CPU `to_float` does.
#![cfg(feature = "rocm")]

use hanzo_ml::backend::{BackendDevice, BackendStorage};
use hanzo_ml::quantized::{ggml_file::qtensor_from_ggml, GgmlDType, QMatMul};
use hanzo_ml::{CpuStorage, DType, Device, Module, Result, RocmQuantType, Tensor};

/// Deterministic bytes: an LCG, so a failure reproduces without a seed file.
struct Lcg(u64);
impl Lcg {
    fn next_u8(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u8
    }
}

/// Every type the unified core wires, less Q8_1: that is the activation format the int8 paths
/// quantize into, no GGML file stores a weight in it, and the CPU loader has no reference for it.
const WIRED: [GgmlDType; 24] = [
    GgmlDType::Q8_0,
    GgmlDType::Q4_0,
    GgmlDType::Q4K,
    GgmlDType::Q6K,
    GgmlDType::IQ4_XS,
    GgmlDType::TQ2_0,
    GgmlDType::Q2K,
    GgmlDType::Q3K,
    GgmlDType::Q5K,
    GgmlDType::Q4_1,
    GgmlDType::Q5_0,
    GgmlDType::Q5_1,
    GgmlDType::IQ2_XXS,
    GgmlDType::IQ2_XS,
    GgmlDType::IQ2_S,
    GgmlDType::IQ3_XXS,
    GgmlDType::IQ3_S,
    GgmlDType::IQ4_NL,
    GgmlDType::TQ1_0,
    GgmlDType::IQ1_S,
    GgmlDType::IQ1_M,
    GgmlDType::MXFP4,
    GgmlDType::ROCMFP4,
    GgmlDType::ROCMFP4_FAST,
];

/// Same value: equal, both NaN, or within a part in 10^5. Random block bytes put every bit
/// pattern in the scale fields, NaN and infinity included, and the device must decode those as
/// the CPU does too. Only the order of a few f32 products differs between the two.
fn same(gpu: f32, cpu: f32) -> bool {
    if gpu.is_nan() || cpu.is_nan() {
        return gpu.is_nan() && cpu.is_nan();
    }
    gpu == cpu || (gpu - cpu).abs() <= 1e-5 * cpu.abs().max(gpu.abs())
}

#[test]
fn every_wired_type_dequantizes_on_the_device_as_the_cpu_does() -> Result<()> {
    let Device::Rocm(rocm) = Device::new_rocm(0)? else {
        unreachable!("new_rocm returns a rocm device")
    };
    let (rows, cols) = (3usize, 512usize);
    let mut rng = Lcg(0xdec0_de00);
    for dtype in WIRED {
        let qt = RocmQuantType::from_ggml(dtype)
            .unwrap_or_else(|| panic!("{dtype:?} has no row in the unified core"));
        let nbytes = rows * cols / qt.block_elems() * qt.block_bytes();
        let raw: Vec<u8> = (0..nbytes).map(|_| rng.next_u8()).collect();

        let want = qtensor_from_ggml(dtype, &raw, vec![rows, cols], &Device::Cpu)?
            .dequantize(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let wq = rocm.storage_from_slice(&raw)?;
        let dense = rocm.dequantize_quant(qt, &wq, rows * cols, DType::F32)?;
        let CpuStorage::F32(got) = dense.to_cpu_storage()? else {
            panic!("{dtype:?}: dequantize_quant(F32) did not return f32")
        };

        assert_eq!(got.len(), want.len(), "{dtype:?}");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(same(*g, *w), "{dtype:?} element {i}: gpu {g} vs cpu {w}");
        }
    }
    Ok(())
}

/// A prompt is a multi-row forward. For a type with no native GEMM that used to dequantize the
/// whole weight on the host, per layer, per forward.
#[test]
fn a_multi_row_forward_of_a_decode_only_type_matches_the_cpu() -> Result<()> {
    let gpu = Device::new_rocm(0)?;
    let (rows, cols, batch) = (96usize, 2048usize, 5usize);
    for dtype in [GgmlDType::ROCMFP4, GgmlDType::ROCMFP4_FAST, GgmlDType::MXFP4] {
        let qt = RocmQuantType::from_ggml(dtype).expect("wired");
        let mut rng = Lcg(0x0ddba11 + dtype as u64);
        let nblocks = rows * cols / qt.block_elems();
        // Nibbles are free; scale bytes stay finite so the 1% gate is about magnitude, not NaN.
        let scales = qt.block_bytes() - 16;
        let mut raw = Vec::with_capacity(nblocks * qt.block_bytes());
        for _ in 0..nblocks {
            match dtype {
                GgmlDType::MXFP4 => {
                    raw.push(120 + rng.next_u8() % 12);
                    raw.extend((0..16).map(|_| rng.next_u8()));
                }
                _ => {
                    raw.extend((0..16).map(|_| rng.next_u8()));
                    raw.extend((0..scales).map(|_| 1 + rng.next_u8() % 0x7e));
                }
            }
        }

        let reference = qtensor_from_ggml(dtype, &raw, vec![rows, cols], &Device::Cpu)?
            .dequantize(&Device::Cpu)?;
        let x: Vec<f32> = (0..batch * cols)
            .map(|_| (rng.next_u8() as f32 - 127.5) / 64.0)
            .collect();
        let x_cpu = Tensor::from_vec(x, (batch, cols), &Device::Cpu)?;
        let want = x_cpu.matmul(&reference.t()?)?.flatten_all()?.to_vec1::<f32>()?;
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(scale > 0.0, "{dtype:?}: degenerate reference");

        let qmatmul =
            QMatMul::from_qtensor(qtensor_from_ggml(dtype, &raw, vec![rows, cols], &gpu)?)?;
        let got = qmatmul
            .forward(&x_cpu.to_device(&gpu)?.to_dtype(DType::F16)?)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(got.len(), want.len(), "{dtype:?}");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 0.01 * scale,
                "{dtype:?} output {i}: gpu {g} vs cpu {w} (gate {})",
                0.01 * scale
            );
        }
    }
    Ok(())
}
