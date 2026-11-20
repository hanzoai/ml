#![cfg(feature = "rocm")]

use half::f16;
use hanzo_ml::quantized::{
    ggml_file::qtensor_from_ggml, moe_combine, moe_gate_up, try_moe_shared_q4k, GgmlDType, QMatMul,
};
use hanzo_ml::{DType, Device, Result, Tensor};

const EXPERTS: usize = 4;
const HIDDEN: usize = 768;
const INTERMEDIATE: usize = 256;
const OUTPUT: usize = 133;
const TOKENS: usize = 65;
const TOP_K: usize = 2;

fn bank(n: usize, k: usize, seed: usize, device: &Device) -> Result<QMatMul> {
    let mut bytes = Vec::with_capacity(EXPERTS * n * (k / 256) * 144);
    for block in 0..EXPERTS * n * (k / 256) {
        // Little-endian FP16 d and dmin, followed by the 12 packed scale/min bytes.
        bytes.extend_from_slice(&[0, 0x20, 0, 0x2b]);
        bytes.extend_from_slice(&[1, 2, 3, 4, 2, 1, 4, 3, 0x31, 0x25, 0x17, 0x43]);
        for lane in 0..128 {
            let lo = (block + lane + seed) % 16;
            let hi = (3 * block + lane / 3 + seed) % 16;
            bytes.push((lo | (hi << 4)) as u8);
        }
    }
    QMatMul::from_qtensor(qtensor_from_ggml(
        GgmlDType::Q4K,
        &bytes,
        vec![EXPERTS, n, k],
        device,
    )?)
}

fn assert_same_bits(stage: &str, expected: &Tensor, actual: &Tensor) -> Result<()> {
    assert_eq!(expected.shape(), actual.shape(), "{stage}");
    assert_eq!(actual.dtype(), DType::F16, "{stage}");
    let expected = expected.flatten_all()?.to_vec1::<f16>()?;
    let actual = actual.flatten_all()?.to_vec1::<f16>()?;
    assert!(
        expected.iter().any(|value| value.to_bits() & 0x7fff != 0),
        "{stage}"
    );
    for (index, (expected, actual)) in expected.into_iter().zip(actual).enumerate() {
        assert_eq!(expected.to_bits(), actual.to_bits(), "{stage} at {index}");
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx1201 hardware and HANZO_ROCM_GFX1201_GROUPED_Q4K=1"]
fn grouped_q4k_matches_indexed_f16() -> Result<()> {
    let device = Device::new_rocm(0)?;
    let gate = bank(INTERMEDIATE, HIDDEN, 1, &device)?;
    let up = bank(INTERMEDIATE, HIDDEN, 5, &device)?;
    let down = bank(OUTPUT, INTERMEDIATE, 9, &device)?;
    let values = (0..TOKENS * HIDDEN)
        .map(|i| [0.125f32, -0.0625, 0.015625, 0.25, -0.125, 0.0][(i + i / HIDDEN) % 6])
        .collect::<Vec<_>>();
    let x = Tensor::from_vec(values, (TOKENS, 1, HIDDEN), &device)?.to_dtype(DType::F16)?;
    let scores = Tensor::from_vec([0.75f32, 0.25].repeat(TOKENS), (TOKENS, TOP_K), &device)?;
    let activate = |g: &Tensor, u: &Tensor| g.silu()? * u;

    // Across multiple planner chunks, one expert has 65 slots, another has a batch and a
    // remainder, a third takes the one-slot tail, and the fourth has no routes.
    let routes = (0..TOKENS).flat_map(|token| [0u32, if token == 7 { 2 } else { 1 }]);
    let ids = Tensor::from_vec(routes.collect::<Vec<_>>(), (TOKENS, TOP_K), &device)?;
    let (reference_gate, reference_up) = moe_gate_up(&x, &ids, &gate, &up)?;
    let reference_down =
        down.indexed_moe_forward(&activate(&reference_gate, &reference_up)?, &ids)?;
    let (grouped_gate, grouped_up, grouped_down) =
        try_moe_shared_q4k(&x, &ids, &gate, &up, &down, activate)?
            .expect("the opt-in gfx1201 kernel must run");
    assert_same_bits("gate", &reference_gate, &grouped_gate)?;
    assert_same_bits("up", &reference_up, &grouped_up)?;
    assert_same_bits("down", &reference_down, &grouped_down)?;
    assert_same_bits(
        "combine",
        &moe_combine(&reference_down, &scores)?,
        &moe_combine(&grouped_down, &scores)?,
    )?;

    let invalid = Tensor::from_vec(vec![0u32, EXPERTS as u32], (2, 1), &device)?;
    let x = x.narrow(0, 0, 2)?;
    let error = try_moe_shared_q4k(&x, &invalid, &gate, &up, &down, |_, _| {
        panic!("invalid expert ID reached the activation")
    })
    .expect_err("out-of-range expert IDs must be rejected");
    assert!(error.to_string().contains("invalid expert ID"));
    Ok(())
}
