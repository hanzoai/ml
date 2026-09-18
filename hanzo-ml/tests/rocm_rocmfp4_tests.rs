//! ROCMFP4 / ROCMFP4_FAST on the ROCm unified quant core, against the CPU reference decode.
//!
//! The GPU reads the GGML blocks straight out of VRAM (`qmatvec_core<WTYPE>`); the CPU path
//! dequantizes the same bytes through `iq_quants`. Both must agree to the core's numeric gate
//! (1% of magnitude — only the f32 accumulation order differs), for both activation dtypes the
//! core instantiates, and the types must ride the native path rather than the dense-f16
//! fallback an unwired type silently takes.
#![cfg(feature = "rocm")]

use hanzo_ml::quantized::{ggml_file::qtensor_from_ggml, GgmlDType, QMatMul};
use hanzo_ml::{DType, Device, Module, Result, RocmQuantType, Tensor};

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

/// `rows * cols` weights as raw blocks: 16 nibble bytes, then `scales` UE4M3 bytes. Scale bytes
/// stay in the valid range (1..=0x7e) except for a sprinkling of the three invalid classes —
/// zero, 0x7f (NaN) and sign-set — which the fork's row validator defines as decoding to zero.
fn blocks(rng: &mut Lcg, rows: usize, cols: usize, scales: usize) -> Vec<u8> {
    let nblocks = rows * cols / 32;
    let mut raw = Vec::with_capacity(nblocks * (16 + scales));
    for b in 0..nblocks {
        for _ in 0..16 {
            raw.push(rng.next_u8());
        }
        for s in 0..scales {
            raw.push(match (b + s) % 61 {
                0 => 0x00,
                1 => 0x7f,
                2 => 0x80 | (rng.next_u8() & 0x7f),
                _ => 1 + rng.next_u8() % 0x7e,
            });
        }
    }
    raw
}

fn parity(dtype: GgmlDType, scales: usize) -> Result<()> {
    assert!(
        RocmQuantType::from_ggml(dtype).is_some(),
        "{dtype:?} has no row in the unified core and would dequantize to dense f16"
    );
    let gpu = Device::new_rocm(0)?;
    let (rows, cols) = (96usize, 2048usize);
    let mut rng = Lcg(0x5eed_f00d);
    let raw = blocks(&mut rng, rows, cols, scales);

    let reference =
        qtensor_from_ggml(dtype, &raw, vec![rows, cols], &Device::Cpu)?.dequantize(&Device::Cpu)?; // [rows, cols] f32
    let x: Vec<f32> = (0..cols)
        .map(|_| (rng.next_u8() as f32 - 127.5) / 64.0)
        .collect();
    let x_cpu = Tensor::from_vec(x, (1, cols), &Device::Cpu)?;
    let want = x_cpu
        .matmul(&reference.t()?)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(scale > 0.0, "degenerate reference: every output is zero");

    let qmatmul = QMatMul::from_qtensor(qtensor_from_ggml(dtype, &raw, vec![rows, cols], &gpu)?)?;
    for act in [DType::F16, DType::BF16] {
        let got = qmatmul
            .forward(&x_cpu.to_device(&gpu)?.to_dtype(act)?)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(got.len(), want.len());
        for (row, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 0.01 * scale,
                "{dtype:?} {act:?} row {row}: gpu {g} vs cpu {w} (gate {})",
                0.01 * scale
            );
        }
    }
    Ok(())
}

#[test]
fn rocmfp4_dual_scale_matches_cpu() -> Result<()> {
    parity(GgmlDType::ROCMFP4, 2)
}

#[test]
fn rocmfp4_fast_matches_cpu() -> Result<()> {
    parity(GgmlDType::ROCMFP4_FAST, 1)
}
