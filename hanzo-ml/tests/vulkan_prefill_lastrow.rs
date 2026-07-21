//! Regression: a quantized matmul whose activation is a NARROWED view (non-zero storage offset, as
//! `extract_logits` produces with `x.narrow(1, seq_len-1, 1)` for the LM head) must read the narrowed
//! rows, not the buffer base. The native Vulkan quant matvec/matmul take the storage with no offset,
//! so QMatMul::forward has to materialize an offset-0 buffer first (see `vulkan_act_offset0`). Before
//! that fix the first generated token's logits were garbage (a stray `:` on Qwen3). Skips with no GPU.
#![cfg(feature = "vulkan")]

use hanzo_ml::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use hanzo_ml::{DType, Device, Module, Tensor};
use std::sync::Arc;

fn gpu() -> Option<Device> {
    match Device::new_vulkan(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("[prefill_lastrow] no Vulkan GPU ({e}); skipping");
            None
        }
    }
}

fn pseudo(i: usize) -> f32 {
    let mut z = (i as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
}

fn to_host(t: &Tensor) -> hanzo_ml::Result<Vec<f32>> {
    t.to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()
}

// max-abs error of a [1,1,H] result vs a CPU f32 reference row.
fn max_err(got: &[f32], reference: &[f32]) -> f32 {
    got.iter()
        .zip(reference)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()))
}

#[test]
fn vulkan_qmatmul_narrowed_activation_matches_cpu() -> hanzo_ml::Result<()> {
    let Some(dev) = gpu() else { return Ok(()) };
    let cpu = Device::Cpu;
    let (s, h) = (15usize, 2048usize); // S == "capital of France" prompt length; H = Qwen3-1.7B hidden

    // Shared f32 input [1, S, H] on both devices.
    let x_host: Vec<f32> = (0..s * h).map(|i| pseudo(i) * 0.5).collect();
    let x_cpu = Tensor::from_vec(x_host.clone(), (1, s, h), &cpu)?;
    let x_vk = Tensor::from_vec(x_host, (1, s, h), &dev)?;

    // One Q4_K weight, identical bytes on both devices.
    let wq_t = Tensor::from_vec((0..h * h).map(|i| pseudo(i + 101) * 0.1).collect(), (h, h), &cpu)?;
    let raw = QTensor::quantize(&wq_t, GgmlDType::Q4K)?.data()?.into_owned();
    let mk = |dev: &Device| -> hanzo_ml::Result<QMatMul> {
        let qs = QStorage::from_data(std::borrow::Cow::Owned(raw.clone()), dev, GgmlDType::Q4K)?;
        QMatMul::from_arc(Arc::new(QTensor::new(qs, (h, h))?))
    };
    let mm_cpu = mk(&cpu)?;
    let mm_vk = mk(&dev)?;

    // CPU f32 reference for the LAST row: dequantize the weight, plain f32 matmul of x[.., last, ..].
    let wdeq = QTensor::new(
        QStorage::from_data(std::borrow::Cow::Owned(raw.clone()), &cpu, GgmlDType::Q4K)?,
        (h, h),
    )?
    .dequantize(&cpu)?;
    let reference = to_host(&x_cpu.narrow(1, s - 1, 1)?.reshape((1, h))?.matmul(&wdeq.t()?)?)?;

    // The exact shape the LM head sees on the prefill path: narrow the last position (non-zero storage
    // offset (S-1)*H), then run the m=1 quant matmul.
    let vk_narrow = to_host(&mm_vk.forward(&x_vk.narrow(1, s - 1, 1)?.contiguous()?)?)?;
    let cpu_narrow = to_host(&mm_cpu.forward(&x_cpu.narrow(1, s - 1, 1)?.contiguous()?)?)?;

    // Sanity: same weights via the full prefill GEMM (offset 0) already match CPU -- so any mismatch is
    // purely the narrowed-activation offset, not quantization.
    let vk_gemm_last = to_host(&mm_vk.forward(&x_vk)?)?[(s - 1) * h..s * h].to_vec();

    let e_vk = max_err(&vk_narrow, &reference);
    let e_cpu = max_err(&cpu_narrow, &reference);
    let e_gemm = max_err(&vk_gemm_last, &reference);
    eprintln!("narrowed m=1: vk_err={e_vk:.3e} cpu_err={e_cpu:.3e}  (gemm_last_err={e_gemm:.3e})");

    // Q4_K quant + q8_1 activation-quant tolerance is ~1e-2; the offset bug produced errors > 1 (larger
    // than the signal). Assert the GPU narrowed matmul tracks CPU within the quant tolerance.
    assert!(
        e_vk < 5e-2,
        "narrowed-activation m=1 Q4_K matmul diverged from CPU: vk_err={e_vk} (offset not honored)"
    );
    Ok(())
}
